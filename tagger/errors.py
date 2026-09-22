class FatalError(RuntimeError):
    """An error that would make every remaining photo fail the same way.

    Raised for problems such as a rejected API key or an unknown model, where
    continuing the run would only repeat the same failing request per photo.
    """
