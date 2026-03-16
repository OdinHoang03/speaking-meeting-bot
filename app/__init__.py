"""Speaking Meeting Bot API package."""


def get_application():
    """Get FastAPI application instance."""
    from app.main import create_app

    return create_app()


# Only create the app when directly accessed, not on import
app = get_application()
