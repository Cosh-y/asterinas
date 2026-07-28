use crate::arch::vm::GuestInterrupt;

/// Provides guest interrupt injection policy to [`super::GuestMode`].
///
/// `GuestMode` uses this port before guest entry to choose whether a pending
/// interrupt should be injected into the guest. If it commits an interrupt for
/// the next entry, it calls [`GuestInterruptPort::accept_interrupt`] so the
/// kernel-side interrupt model can synchronize its state.
///
/// The implementation is supplied by the kernel. It may model a virtual
/// interrupt controller, or it may be a policy object that never offers
/// interrupts.
///
/// These methods may be called while guest entry preparation has disabled
/// preemption or local interrupts. Implementations must not sleep, yield, or
/// wait on synchronization primitives that can block the current task.
pub trait GuestInterruptPort {
    /// Returns the next guest interrupt to offer for injection.
    ///
    /// This method is a query. It should not consume the interrupt because
    /// `GuestMode` may find that the guest cannot accept it yet.
    /// Returning `None` means that no interrupt should be offered for this
    /// guest entry.
    ///
    /// An implementation that does not inject guest interrupts can always
    /// return `None`.
    fn query_pending_interrupt(&self) -> Option<GuestInterrupt>;

    /// Marks a guest interrupt as accepted for injection.
    ///
    /// `GuestMode` calls this method only after it has committed the interrupt
    /// for the next guest entry. Implementations should update their state
    /// accordingly.
    fn accept_interrupt(&self, interrupt: GuestInterrupt);
}
