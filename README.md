# Indian Cattle & Buffalo Recognition System

An AI-powered web application for identifying breeds and estimating ages of Indian cattle and buffaloes using deep learning.

## Features

- **Breed Classification**: Identifies 19 Indian cattle and buffalo breeds
  - Cattle: Gir, Sahiwal, Red Sindhi, Ongole, Tharparkar, Kankrej, Hariana, Rathi, Krishna Valley, Deoni
  - Buffalo: Murrah, Jaffarabadi, Surti, Mehsana, Bhadawari, Nili-Ravi, Pandharpuri, Nagpuri, Toda

- **Age Estimation**: Classifies into 4 age groups
  - Calf (0-6 months)
  - Young (6-18 months)
  - Adult (18 months - 6 years)
  - Old (6+ years)

- **Modern Web Interface**: Clean, mobile-friendly UI with drag-and-drop upload
- **Real-time Predictions**: Instant results with confidence scores
- **Prediction History**: Database storage of all predictions
- **Admin Panel**: Manage breeds and view statistics
- **REST API**: Full API documentation with Swagger/OpenAPI

## Tech Stack

- **Backend**: FastAPI (Python)
- **Deep Learning**: PyTorch with MobileNetV2 backbone
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: SQLite with SQLAlchemy
- **Image Processing**: OpenCV, Pillow

## Installation

1. **Clone or download the project**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python run.py
   ```

5. **Access the application**:
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Admin Panel: http://localhost:8000/admin

## Project Structure

```
age and breed/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py    # CNN model definition
│   │   └── preprocessing.py # Image preprocessing
│   ├── routers/
│   │   ├── __init__.py
│   │   └── prediction.py    # API endpoints
│   ├── database/
│   │   ├── __init__.py
│   │   └── db.py           # Database models
│   └── schemas/
│       ├── __init__.py
│       └── prediction.py   # Pydantic schemas
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   ├── index.html          # Main interface
│   └── admin.html          # Admin panel
├── saved_models/           # Trained model weights
├── uploads/                # Uploaded images
├── requirements.txt
├── run.py                  # Application runner
├── train.py               # Model training script
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Upload image and get predictions |
| `/api/predictions` | GET | Get prediction history |
| `/api/predictions/{id}` | GET | Get specific prediction |
| `/api/feedback` | POST | Submit feedback on predictions |
| `/api/statistics` | GET | Get system statistics |
| `/api/breeds` | GET | Get list of supported breeds |
| `/api/age-groups` | GET | Get list of age groups |
| `/api/admin/breeds` | GET/POST | Manage breed information |

## Training Your Own Model

1. **Prepare your dataset**:
   ```
   data/
   ├── images/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── labels.json
   ```

2. **Create labels.json** with format:
   ```json
   {
     "image1.jpg": {
       "breed": "Gir",
       "age": "Adult (18 months - 6 years)",
       "type": "Cattle"
     }
   }
   ```

3. **Run training**:
   ```bash
   python train.py --data data --epochs 50 --batch-size 32
   ```

4. **Create sample labels file**:
   ```bash
   python train.py --create-sample
   ```

## Configuration

The application uses default settings that work out of the box. For production:

- Set up a proper database (PostgreSQL recommended)
- Configure CORS for your domain
- Use a production WSGI server (gunicorn)
- Set up SSL/TLS

## Notes

- The model uses transfer learning with MobileNetV2 pretrained on ImageNet
- Without custom training data, predictions use the pretrained backbone with random classification heads
- For accurate predictions, train the model on a labeled dataset of Indian cattle and buffalo images

## License

This project is for educational and research purposes.

## Acknowledgments

- Built with FastAPI and PyTorch
- Uses pretrained MobileNetV2 weights from torchvision
