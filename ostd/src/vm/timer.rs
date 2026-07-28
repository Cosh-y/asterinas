use crate::arch::vm::GuestTimerInstant;

/// Provides guest timer interrupt policy to [`super::GuestMode`].
///
/// `GuestMode` uses this port before VM entry to ask when a VM exit should
/// happen so the kernel can publish a virtual timer interrupt in time.
///
/// This method may be called while guest entry preparation has disabled
/// preemption or local interrupts. Implementations must not sleep, yield, or
/// wait on synchronization primitives that can block the current task.
pub trait GuestTimerPort {
    /// Returns the next guest timer deadline after `current`.
    ///
    /// The `current` argument is the current guest-visible timer instant. The
    /// returned deadline is expressed on the same guest-visible timeline.
    ///
    /// Returning `Some(deadline)` asks OSTD to arrange a VM exit when that
    /// deadline is reached.
    fn poll_deadline(&self, current: GuestTimerInstant) -> Option<GuestTimerInstant>;
}
