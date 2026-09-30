from __future__ import annotations

from dataclasses import dataclass

from fabweave.core.ids import deterministic_id
from fabweave.core.quantities import require_conservation
from fabweave.lot.merge import Lot


@dataclass(frozen=True, slots=True)
class MachineCapacity:
    """Idle capacity on one parallel machine plus the reticle planned for it."""

    tool_id: str
    chamber_id: str
    reticle_id: str
    available_units: int


@dataclass(frozen=True, slots=True)
class ChildLot:
    """One split child: parent route metadata carried forward, machine bound."""

    lot_id: str
    parent_lot_id: str
    product: str
    route: str
    step: str
    recipe_revision: int
    tool_id: str
    chamber_id: str
    reticle_id: str
    quantity: int


def _even_shares(quantity: int, capacities: list[int]) -> list[int] | None:
    """Split ``quantity`` as evenly as possible without exceeding any capacity."""
    pinned: dict[int, int] = {}
    open_indexes = list(range(len(capacities)))
    while open_indexes:
        remaining = quantity - sum(pinned.values())
        share, extra = divmod(remaining, len(open_indexes))
        tight = [index for index in open_indexes if capacities[index] <= share]
        if not tight:
            shares = dict(pinned)
            for index in open_indexes:
                shares[index] = share
            for index in open_indexes[:extra]:
                shares[index] += 1
            return [shares[index] for index in range(len(capacities))]
        for index in tight:
            pinned[index] = capacities[index]
        open_indexes = [index for index in open_indexes if capacities[index] > share]
    if sum(pinned.values()) == quantity:
        return [pinned[index] for index in range(len(capacities))]
    return None


def split_allocation(quantity: int, capacities: list[int], min_child_units: int) -> tuple[tuple[int, int], ...]:
    """Spread ``quantity`` across machines as evenly as their capacities allow.

    Returns ``(machine_index, units)`` pairs for the machines that receive a
    child.  Every share respects the machine's available capacity and the
    minimum child size; when the minimum cannot be met on all eligible
    machines, the smallest-capacity machine is dropped and the split is
    retried, preferring the widest parallel spread.  Raises when no feasible
    split into at least two children exists.
    """
    if min_child_units < 1:
        raise ValueError("minimum child size must be positive")
    if quantity < 1:
        raise ValueError("quantity must be positive")
    if any(capacity < 0 for capacity in capacities):
        raise ValueError("capacity must be non-negative")
    eligible = [index for index, capacity in enumerate(capacities) if capacity >= min_child_units]
    while len(eligible) >= 2:
        if sum(capacities[index] for index in eligible) < quantity:
            break
        shares = _even_shares(quantity, [capacities[index] for index in eligible])
        if shares is not None and min(shares) >= min_child_units:
            return tuple(zip(eligible, shares))
        smallest = min(range(len(eligible)), key=lambda position: (capacities[eligible[position]], -position))
        eligible.pop(smallest)
    raise ValueError("insufficient machine capacity for a split at the minimum size")


def plan_split(request_id: str, lot: Lot, machines: list[MachineCapacity], min_child_units: int) -> tuple[ChildLot, ...]:
    """Plan a capacity-driven split; pure and deterministic, raises on any violation."""
    if lot.active_holds:
        raise ValueError("held lots cannot split")
    if lot.quantity < 2 * min_child_units:
        raise ValueError("lot quantity cannot yield two children at the minimum size")
    if len(machines) < 2:
        raise ValueError("a split requires at least two machines")
    identities = [(machine.tool_id, machine.chamber_id) for machine in machines]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate machine capacity")
    ordered = sorted(machines, key=lambda machine: (machine.tool_id, machine.chamber_id))
    pieces = split_allocation(lot.quantity, [machine.available_units for machine in ordered], min_child_units)
    children = []
    for index, units in pieces:
        machine = ordered[index]
        child_id = deterministic_id("LOT", request_id, lot.lot_id, machine.tool_id, machine.chamber_id)
        children.append(
            ChildLot(
                lot_id=child_id,
                parent_lot_id=lot.lot_id,
                product=lot.product,
                route=lot.route,
                step=lot.step,
                recipe_revision=lot.recipe_revision,
                tool_id=machine.tool_id,
                chamber_id=machine.chamber_id,
                reticle_id=machine.reticle_id,
                quantity=units,
            )
        )
    require_conservation({lot.lot_id: lot.quantity}, {child.lot_id: child.quantity for child in children})
    return tuple(children)


class SplitLedger:
    """Committed splits: idempotent by request, one split per parent lot."""

    def __init__(self) -> None:
        self._by_request: dict[str, tuple[ChildLot, ...]] = {}
        self._by_parent: dict[str, tuple[ChildLot, ...]] = {}

    def by_request(self, request_id: str) -> tuple[ChildLot, ...] | None:
        return self._by_request.get(request_id)

    def already_split(self, lot_id: str) -> bool:
        return lot_id in self._by_parent

    def children_of(self, lot_id: str) -> tuple[ChildLot, ...]:
        return self._by_parent.get(lot_id, ())

    def record(self, request_id: str, plan: tuple[ChildLot, ...]) -> tuple[ChildLot, ...]:
        previous = self._by_request.get(request_id)
        if previous is not None:
            if previous != plan:
                raise ValueError("request id reused with different split")
            return previous
        if not plan:
            raise ValueError("empty split plan")
        parents = {child.parent_lot_id for child in plan}
        if len(parents) != 1:
            raise ValueError("split plan must reference exactly one parent")
        parent = next(iter(parents))
        if parent in self._by_parent:
            raise ValueError("parent lot already split")
        child_ids = [child.lot_id for child in plan]
        if len(set(child_ids)) != len(child_ids):
            raise ValueError("duplicate child lot id")
        frozen = tuple(plan)
        self._by_request[request_id] = frozen
        self._by_parent[parent] = frozen
        return frozen
