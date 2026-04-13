pub(crate) type Result<T> = core::result::Result<T, Error>;

/// The error types used in this crate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Errno {
    InvalidArgs,
    Fault,
    /// Error from lower ostd
    OstdError,
    /// resource not found
    NotFound,
    /// Guest's vcpu run failed
    GuestRunFailed,
}

/// The error with an error type and an error message used in this crate.
#[derive(Clone, Debug)]
pub struct Error {
    errno: Errno,
    msg: Option<&'static str>,
}

impl Error {
    pub fn with_message(err: Errno, msg: &'static str) -> Self {
        Error {
            errno: err,
            msg: Some(msg),
        }
    }

    pub fn errno(&self) -> Errno {
        self.errno
    }

    pub fn message(&self) -> Option<&'static str> {
        self.msg
    }
}

impl From<ostd::Error> for Error {
    fn from(value: ostd::Error) -> Self {
        match value {
            ostd::Error::AccessDenied => Self {
                errno: Errno::OstdError,
                msg: Some("ostd error: AccessDenied"),
            },
            ostd::Error::InvalidArgs => Self {
                errno: Errno::OstdError,
                msg: Some("ostd error: InvalidArgs"),
            },
            _ => Self {
                errno: Errno::OstdError,
                msg: None,
            }, // Add more categories
        }
    }
}
