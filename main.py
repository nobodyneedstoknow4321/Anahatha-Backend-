from fastapi import FastAPI, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import bcrypt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
heart_data = pd.read_csv('heart_disease_data.csv')
app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Database connection
conn = psycopg2.connect(
    host="localhost",
    database="Anahatha",
    user="postgres",
    password="test"
)

# User model


class User(BaseModel):
    email: str
    password: str
    age: int
    sex: str
    name: str

# HeartDisease model


class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# LoginUser model


class LoginUser(BaseModel):
    email: str
    password: str


class Post(BaseModel):
    email: str
    title: str
    body: str
    post_id: Optional[int]
# Signup endpoint


class Comment(BaseModel):

    name: str
    role: str
    body: str
    post_id: Optional[int]
    comment_id: Optional[int]
    replies_count: Optional[int] = 0


class Doctor(BaseModel):
    name: str
    medical_id: int
    college: Optional[str]


class Reply(BaseModel):
    name: str
    role: str
    body: str
    post_id: Optional[int]
    comment_id: Optional[int]
    reply_id: Optional[int]


class CheckDoctor(BaseModel):
    email: str
    medical_id: str
    name: str


@app.post("/signup")
def signup(user: User):
    cursor = conn.cursor()

    # Check if email already exists
    cursor.execute("SELECT email FROM users WHERE email=%s", (user.email,))
    existing_user = cursor.fetchone()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password
    hashed_password = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())

    # Insert user into database
    cursor.execute(
        "INSERT INTO users (email, password, age, sex, name) VALUES (%s, %s, %s, %s, %s)",
        (user.email, hashed_password.decode(), user.age, user.sex, user.name)
    )
    conn.commit()
    user_data = {
        "email": user.email,
        "password": user.password,
        "age": user.age,
        "sex": user.sex,
        "name": user.name
    }
    return {"data": user_data}

# Login endpoint


@app.post("/login")
def login(user: LoginUser):
    cursor = conn.cursor()

    # Retrieve user from database
    cursor.execute(
        "SELECT email, password, age, sex, name,doctor FROM users WHERE email=%s", (user.email,))
    db_user = cursor.fetchone()

    if db_user is None:
        raise HTTPException(
            status_code=400, detail="Incorrect email or password")

    # Verify password
    if bcrypt.checkpw(user.password.encode(), db_user[1].encode()):
        # Return user data
        return {
            "email": db_user[0],
            "age": db_user[2],
            "sex": db_user[3],
            "name": db_user[4],
            "doctor": db_user[5],
        }
    else:
        raise HTTPException(
            status_code=400, detail="Incorrect email or password")


@app.post("/predict")
def predict(data: HeartDiseaseInput):
    # Splitting the Features and the Target
    X = heart_data.drop(columns="target", axis=1)
    Y = heart_data["target"]

    # Splitting data into testing and train data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Make the prediction
    input_data = np.asarray(list(data.dict().values()))
    input_data = input_data.astype(int)
    input_data_reshaped = input_data.reshape(1, -1)
    input_data_reshaped = scaler.transform(input_data_reshaped)
    prediction = model.predict(input_data_reshaped)

    # Insert a new row into the testreports table
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO testreports (
            age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal, email, prediction
        )
        VALUES (
             %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s::integer
        );
    """, (
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak,
        data.slope, data.ca, data.thal, "test@gmail.com", int(prediction[0])

    ))
    conn.commit()
    cursor.close()

    # Return the result
    if prediction[0] == 1:
        return {"result": "Heart Defect"}
    else:
        return {"result": "No Heart Defect"}


@app.post("/post")
def create_post(post: Post):
    def generate_post_id():
        while True:
            post_id = random.randint(100000, 999999)
            cursor.execute(
                "SELECT post_id FROM posts WHERE post_id = %s", (post_id,))
            if cursor.fetchone() is None:
                return post_id

    cursor = conn.cursor()
    post_id = generate_post_id()
    cursor.execute("""INSERT INTO posts (email,title, body, post_id) VALUES (%s,%s, %s, %s);""",
                   (post.email, post.title, post.body, post_id))
    conn.commit()
    cursor.close()
    return {"data": post}


@app.get("/post")
def get_all_posts():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posts")
    posts = cursor.fetchall()
    cursor.close()

    return {"data": posts}


@app.get("/post/{post_id}")
def get_post_by_id(post_id: int):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posts WHERE post_id = %s", (post_id,))
    post = cursor.fetchone()
    cursor.close()
    if post:
        return {"data": post}
    else:
        return {"message": "Post not found"}


@app.delete("/post/{post_id}")
def delete_post(post_id: int):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM posts WHERE post_id = %s", (post_id,))
    conn.commit()
    cursor.close()
    return {"message": f"Post {post_id} deleted successfully"}


@app.post("/post/{post_id}/comment")
def create_comment(post_id: int, comment: Comment):

    def generate_comment_id():
        while True:
            comment_id = random.randint(100000, 999999)
            cursor.execute(
                "SELECT comment_id FROM comments WHERE comment_id = %s", (comment_id,))
            if cursor.fetchone() is None:
                return comment_id

    cursor = conn.cursor()
    comment_id = generate_comment_id()
    dummy_comment = cursor.execute("""INSERT INTO comments(name,role,body,post_id,comment_id) VALUES (%s,%s, %s, %s,%s);""", (
        comment.name, comment.role, comment.body, post_id, comment_id))
    conn.commit()
    cursor.close()
    return {"data": dummy_comment}


@app.get("/post/{post_id}/comment")
def get_all_comments(post_id: int):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM comments WHERE post_id = %s", (post_id,))
    comments = cursor.fetchall()
    cursor.close()
    if comments:
        return {"data": comments}
    else:
        return {"message": "No comments found for post"}


@app.get("/post/{post_id}/comment/{comment_id}")
def get_comment(post_id: int, comment_id: int):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM comments WHERE post_id = %s AND comment_id = %s", (post_id, comment_id))
    comment = cursor.fetchone()
    cursor.close()
    if comment:
        return {"data": comment}
    else:
        return {"message": "Comment not found"}


@app.post("/post/{post_id}/comment/{comment_id}/reply")
def create_reply(post_id: int, comment_id: int, reply: Reply):
    def generate_reply_id():
        while True:
            reply_id = random.randint(100000, 999999)
            cursor.execute(
                "SELECT reply_id FROM replies WHERE reply_id = %s", (reply_id,))
            if cursor.fetchone() is None:
                return reply_id
    cursor = conn.cursor()
    reply_id = generate_reply_id()
    dummy_reply = cursor.execute("""INSERT INTO replies(name,role,body,post_id,comment_id,reply_id) VALUES (%s,%s, %s, %s,%s,%s);""", (
        reply.name, reply.role, reply.body, post_id, comment_id, reply_id))
    conn.commit()
    cursor.close()
    return {"data": dummy_reply}
# @app.post("/post/{post_id}/comment/{comment_id}")


# @app.get("post/{post_id}/comment/{comment_id}/reply")
# def get_reply(post_id: int, comment_id: int):
#     cursor = conn.cursor()
#     cursor.execute(
#         """SELECT * FROM replies WHERE post_id = %s AND comment_id= %s""", (post_id, comment_id))
#     replies = cursor.fetchall()
#     cursor.close()
#     if replies:
#         return {"data": replies}
#     else:
#         return {"message": "No comments found for post"}
@app.get("/post/{post_id}/comment/{comment_id}/reply/{reply_id}")
def get_reply(post_id: int, comment_id: int, reply_id: int):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, role, body, post_id, comment_id, reply_id
        FROM replies
        WHERE post_id = %s AND comment_id = %s AND reply_id = %s
    """, (post_id, comment_id, reply_id))
    reply_data = cursor.fetchone()
    cursor.close()
    if reply_data is not None:
        return {"data": reply_data}


@app.get("/post/{post_id}/comment/{comment_id}/reply")
def get_all_replies(post_id: int, comment_id: int):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, role, body, post_id, comment_id, reply_id
        FROM replies
        WHERE post_id = %s AND comment_id = %s
    """, (post_id, comment_id))
    reply_data = cursor.fetchall()
    cursor.close()

    return {"data": reply_data}


@app.post("/doctor")
def enter_doc(doctor: Doctor):
    cursor = conn.cursor()
    cursor.execute("""INSERT INTO doctor(name,medical_id,college) VALUES(%s,%s,%s);""", (
        doctor.name, doctor.medical_id, doctor.college
    ))
    conn.commit()
    cursor.close()
    return {"data": "Doctor Reborn"}


@app.put("/updateDoctor")
async def update_doctor(check: CheckDoctor):
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM doctor WHERE medical_id = %s AND name = %s", (check.medical_id, check.name))
    result = cursor.fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail="Doctor not found")

    cursor.execute(
        "UPDATE users SET doctor = 'true' WHERE email = %s", (check.email,))
    conn.commit()

    cursor.close()

    return {"message": f"User with email {check.email} role updated to Doctor"}
