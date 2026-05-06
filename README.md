## Related Repository

The complete project including both frontend and backend source code is available at:

https://github.com/sahib8508/feedback-ranker-both-frontend-and-backend-


# Feedback Ranker

A full-stack web application that enables students to submit verified feedback about their colleges, and allows college administrators to manage, respond to, and analyze that feedback through an AI-powered dashboard.

---

## Overview

Feedback Ranker addresses a real gap in college accountability by giving students a structured, verified channel to share their experiences. Each submission is tied to a verified student identity, ensuring authenticity. College administrators receive a dedicated dashboard to manage feedback, communicate with students via email, and access AI-generated summaries of overall sentiment.

The platform also includes a college resource sharing system and a library book request module, making it a comprehensive support tool for both students and institutions.

---

## Features

**Student Side**
- Roll number verification against college-uploaded CSV data before feedback submission
- Feedback submission with department, year, and category tagging
- College browsing with real feedback statistics and AI summaries
- Library book request submission
- College resource downloads including notes, papers, and documents

**College Dashboard**
- Secure login for registered college organizations
- View, resolve, and reject student feedback
- Email students directly from the dashboard with custom responses
- AI-generated summaries of student feedback using HuggingFace
- Upload and manage academic resources for students
- Library request management with email notifications

**Super Admin Panel**
- Approve or reject college registration requests
- View and manage all registered colleges
- Monitor platform-wide activity

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, React Router v7 |
| Backend | Python 3, Flask |
| Database | Firebase Firestore |
| File Storage | Firebase Storage |
| Authentication | Firebase Auth |
| AI Summarization | HuggingFace Inference API (distilbart-cnn-12-6) |
| Email | Gmail SMTP via Python smtplib |
| Sentiment Analysis | TextBlob |
| Styling | CSS Modules, react-icons |

---

## Repository Structure

```
Feedback Ranker/
├── backend/
│   ├── app.py                  # Main Flask application
│   ├── firebase_key.json       # Firebase service account (not committed)
│   ├── requirements.txt        # Python dependencies
│   ├── Procfile                # Process declaration for deployment
│   └── .env                    # Environment variables (not committed)
│
└── frontend/
    ├── public/
    └── src/
        ├── App.js
        ├── LandingPage.js
        ├── OrganizationLogin.js
        ├── OrganizationRegister.js
        ├── CollegeDashboard.js
        ├── SuperAdminPanel.js
        ├── StudentOptions.js
        ├── StudentFeedbackSubmission.js
        ├── ReviewColleges.js
        ├── CollegeResources.js
        ├── LibraryBookRequest.js
        ├── FeedbackSummary.js
        ├── firebaseConfig.js
        ├── apiConfig.js
        ├── contentValidator.js
        ├── DarkModeContext.js
        └── UnderConstruction.js
```

---

## Getting Started

### Prerequisites

- Node.js v18 or higher
- Python 3.10 or higher
- A Firebase project with Firestore and Storage enabled
- A HuggingFace account with an API token
- A Gmail account with an App Password configured

### Backend Setup

1. Navigate to the backend directory:

```bash
cd backend
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the backend directory with the following variables:

```
HUGGINGFACE_API_KEY=your_huggingface_token

FLASK_APP=app.py
FLASK_ENV=development
PORT=5000

FIREBASE_PROJECT_ID=your_project_id
FIREBASE_PRIVATE_KEY_ID=your_private_key_id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=your_service_account_email
FIREBASE_CLIENT_ID=your_client_id
FIREBASE_CLIENT_CERT_URL=your_cert_url

EMAIL_SENDER=your_gmail_address
EMAIL_PASSWORD=your_gmail_app_password
```

5. Start the Flask server:

```bash
python app.py
```

The backend will run on `http://localhost:5000`.

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Ensure `src/apiConfig.js` points to your local backend:

```javascript
const API_BASE_URL = "http://localhost:5000";
export default API_BASE_URL;
```

4. Start the React development server:

```bash
npm start
```

The frontend will run on `http://localhost:3000`.

---

## Environment Variables Reference

| Variable | Description |
|---|---|
| `HUGGINGFACE_API_KEY` | Token from huggingface.co/settings/tokens |
| `FIREBASE_PROJECT_ID` | Firebase project identifier |
| `FIREBASE_PRIVATE_KEY` | Private key from Firebase service account JSON |
| `FIREBASE_PRIVATE_KEY_ID` | Key ID from service account JSON |
| `FIREBASE_CLIENT_EMAIL` | Service account email |
| `FIREBASE_CLIENT_ID` | Service account client ID |
| `FIREBASE_CLIENT_CERT_URL` | Certificate URL from service account JSON |
| `EMAIL_SENDER` | Gmail address used to send notifications |
| `EMAIL_PASSWORD` | Gmail App Password (not your account password) |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Detailed service status |
| POST | `/verify_student` | Verify student roll number against CSV |
| POST | `/submit_feedback` | Submit student feedback |
| GET | `/api/dashboard_data` | Fetch feedback and library data for a college |
| GET | `/api/college_stats` | Fetch all approved colleges with feedback statistics |
| POST | `/api/ai_feedback_summary` | Generate AI summary of feedback |
| POST | `/send_feedback_response` | Send email response and update feedback status |
| GET | `/api/colleges` | List approved colleges |
| GET | `/api/resources` | Fetch uploaded resources with filters |
| POST | `/api/resources/upload` | Upload a new resource file |
| GET | `/api/resources/download/<id>` | Download a resource by ID |
| POST | `/request_book` | Submit a library book request |
| POST | `/mark_book_available` | Mark a book as available and notify student |

---

## AI Summarization

The platform uses the `sshleifer/distilbart-cnn-12-6` model via the HuggingFace Inference API to summarize collections of student feedback. The system includes a three-level fallback:

1. HuggingFace Inference API call with `X-Wait-For-Model` header to handle cold starts
2. Local keyword-based analysis using sentiment and topic detection
3. Basic count-based fallback summary

This ensures the application remains functional even if the external API is unavailable.

---

## Firebase Setup

The application requires the following Firestore collections:

| Collection | Purpose |
|---|---|
| `colleges` | College registration and status |
| `feedbacks` | Student feedback submissions |
| `libraryRequests` | Library book requests |
| `resources` | Academic resource metadata |

Firebase Storage is used to store uploaded resource files and college student CSV data for roll number verification.

---

## Security Notes

- Student feedback submission requires prior roll number verification against a college-uploaded CSV
- All college dashboard operations are scoped to the authenticated college code
- Super admin operations are restricted to verified admin credentials
- Firebase private keys and API tokens must never be committed to version control
- The `.env` file and `firebase_key.json` are excluded via `.gitignore`

---



---

## Author

Sahib Hussain
