# Nexus Admin Dashboard

A Flask + React admin dashboard with **JWT + Refresh Token** authentication and **Role-Based Access Control (RBAC)**.

---

## Tech Stack

| Layer    | Technology                                     |
|----------|------------------------------------------------|
| Backend  | Python 3.10+, Flask 3, Flask-SQLAlchemy        |
| Auth     | Flask-JWT-Extended (access + refresh tokens)   |
| Database | SQLite (via SQLAlchemy ORM)                    |
| Frontend | React 18 (CDN), Babel Standalone, Axios        |

---

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The server starts on **http://localhost:5000** and seeds two demo users:

| Username | Password  | Role   |
|----------|-----------|--------|
| admin    | admin123  | Admin  |
| viewer   | viewer123 | Viewer |

### 2. Frontend

Open `frontend/index.html` directly in a browser (no build step needed).

---

## Architecture

### JWT Flow

```
Login → access_token (15 min) + refresh_token (7 days)
         │
         ├─ access_token  → stored in memory (never localStorage)
         └─ refresh_token → stored in localStorage
                             │
                             └─ On 401 TOKEN_EXPIRED → auto-refresh silently
                                On refresh failure   → redirect to login
```

### Token Security
- **Access token** lives only in JS memory — not persisted to localStorage/sessionStorage
- **Refresh token** in localStorage — survives page reloads
- **Logout** adds the JTI to a `token_blocklist` table → immediate server-side revocation
- **Concurrent refresh** is de-duplicated: only one refresh call runs even if multiple requests fail simultaneously

### RBAC

| Role   | Dashboard | View Users | Create/Edit/Delete Users |
|--------|-----------|------------|--------------------------|
| Admin  | ✓         | ✓          | ✓                        |
| Viewer | ✓         | ✓          | ✗ (403 Forbidden)        |

The `@admin_required` decorator on backend routes enforces this independently of UI restrictions.

---

## API Endpoints

### Auth
| Method | Endpoint            | Auth    | Description              |
|--------|---------------------|---------|--------------------------|
| POST   | /api/auth/login     | None    | Login, get tokens        |
| POST   | /api/auth/refresh   | Refresh | Get new access token     |
| DELETE | /api/auth/logout    | Access  | Revoke access token      |
| GET    | /api/auth/me        | Access  | Current user info        |

### Dashboard
| Method | Endpoint                | Auth   | Roles        |
|--------|-------------------------|--------|--------------|
| GET    | /api/dashboard/stats    | Access | All roles    |

### Users
| Method | Endpoint              | Auth   | Roles       |
|--------|-----------------------|--------|-------------|
| GET    | /api/users            | Access | All roles   |
| POST   | /api/users            | Access | Admin only  |
| GET    | /api/users/:id        | Access | All roles   |
| PUT    | /api/users/:id        | Access | Admin only  |
| DELETE | /api/users/:id        | Access | Admin only  |

### Roles
| Method | Endpoint    | Auth   | Description   |
|--------|-------------|--------|---------------|
| GET    | /api/roles  | Access | List all roles |

---

## Production Hardening Checklist

- [ ] Set `JWT_SECRET_KEY` via environment variable (never hardcode)
- [ ] Switch to PostgreSQL for production (`DATABASE_URL` env var)
- [ ] Enable HTTPS (TLS) — never run JWT auth over plain HTTP
- [ ] Restrict CORS origins to your frontend domain
- [ ] Set `DEBUG=False` in production
- [ ] Add rate limiting on `/api/auth/login` to prevent brute force
- [ ] Rotate JWT secret periodically

---

## Environment Variables

```
JWT_SECRET_KEY=your-super-secret-key-here
DATABASE_URL=postgresql://user:pass@host:5432/dbname  # optional, defaults to SQLite
```
