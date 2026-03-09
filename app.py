from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from typing import Optional

# Project imports
from src.constants import APP_HOST, APP_PORT
from src.pipeline.training_pipeline import TrainPipeline
from src.pipeline.prediction_pipeline import ETAData, ETAPredictor

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# FORM DATA HANDLER
# =====================================================
class ETAForm:
    def __init__(self, request: Request):
        self.request = request

        self.delivery_partner: Optional[str] = None
        self.package_type: Optional[str] = None
        self.vehicle_type: Optional[str] = None
        self.delivery_mode: Optional[str] = None
        self.region: Optional[str] = None
        self.weather_condition: Optional[str] = None
        self.distance_km: Optional[float] = None
        self.package_weight_kg: Optional[float] = None

    async def get_form_data(self):
        form = await self.request.form()

        self.delivery_partner = form.get("delivery_partner")
        self.package_type = form.get("package_type")
        self.vehicle_type = form.get("vehicle_type")
        self.delivery_mode = form.get("delivery_mode")
        self.region = form.get("region")
        self.weather_condition = form.get("weather_condition")
        self.distance_km = float(form.get("distance_km", 0))
        self.package_weight_kg = float(form.get("package_weight_kg", 0))

# =====================================================
# ROUTES
# =====================================================

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "eta.html",
        {"request": request, "context": "Enter delivery details"}
    )


@app.get("/train")
async def train():
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        return Response(f"Training failed: {e}")


# Health check endpoint (used by Docker)
@app.get("/health")
async def health():
    return {"status": "running"}


@app.post("/")
async def predict(request: Request):
    try:
        form = ETAForm(request)
        await form.get_form_data()

        eta_data = ETAData(
            delivery_partner=form.delivery_partner,
            package_type=form.package_type,
            vehicle_type=form.vehicle_type,
            delivery_mode=form.delivery_mode,
            region=form.region,
            weather_condition=form.weather_condition,
            distance_km=form.distance_km,
            package_weight_kg=form.package_weight_kg
        )

        input_df = eta_data.get_input_dataframe()
        predictor = ETAPredictor()

        prediction = float(predictor.predict(input_df)[0])

        if prediction < 0:
            message = f"Delivery expected {abs(prediction):.2f} minutes EARLY"
        else:
            message = f"Delivery expected {prediction:.2f} minutes LATE"

        return templates.TemplateResponse(
            "eta.html",
            {
                "request": request,
                "context": message
            }
        )

    except Exception as e:
        return {"error": str(e)}

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)