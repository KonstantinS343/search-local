from fastapi import FastAPI

from routers.default_routers import echo_router, config_router
from routers.sync_routers import sync_router
from routers.user_routers import user_router
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.include_router(echo_router)
app.include_router(config_router)
app.include_router(sync_router)
app.include_router(user_router)

app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == '__main__':
    import uvicorn  

    uvicorn.run(app="main:app", host='fastapi', port=2000, reload=True)