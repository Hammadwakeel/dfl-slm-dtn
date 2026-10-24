import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from app.core.config import DeviceConfig, Settings
from app.db.models import Base
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_name="test dtn",
        host="127.0.0.1",
        port=8000,
        api_token="test-dtn-token-that-is-long-and-random",
        allowed_origins=("http://localhost:5173",),
        database_url="sqlite://",
        nodes=(
            DeviceConfig(
                node_id="master",
                hostname="dfl-master",
                ip="192.168.11.15",
                role="master",
                port=5002,
                token="test-master-token",
            ),
            DeviceConfig(
                node_id="worker1",
                hostname="dfl-worker1",
                ip="192.168.11.123",
                role="worker",
                port=5001,
                token="test-worker-token",
            ),
            DeviceConfig(
                node_id="worker2",
                hostname="dfl-worker2",
                ip="192.168.11.14",
                role="worker",
                port=5001,
                token="test-worker-token",
            ),
            DeviceConfig(
                node_id="worker3",
                hostname="dfl-worker3",
                ip="192.168.11.13",
                role="worker",
                port=5001,
                token="test-worker-token",
            ),
        ),
        poll_interval_seconds=3,
        request_timeout_seconds=1,
        history_retention_count=100,
        master_slm_url="http://127.0.0.1:5003",
        master_slm_token="test-master-slm-token-that-is-long-and-random",
        slm_timeout_seconds=1,
    )


@pytest.fixture
def client(settings: Settings):
    app = create_app(settings, run_poller=False, run_migrations=False)
    engine = app.state.engine
    Base.metadata.create_all(engine)
    from app.db.models import Device

    with app.state.session_factory() as session:
        for config in settings.nodes:
            session.add(
                Device(
                    node_id=config.node_id,
                    role=config.role,
                    hostname=config.hostname,
                    ip=config.ip,
                    port=config.port,
                    online=False,
                )
            )
        session.commit()
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


@pytest.fixture
def auth_headers(settings: Settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.api_token}"}
