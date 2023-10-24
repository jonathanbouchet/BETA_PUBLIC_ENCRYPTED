"""Module for handling authentication, interactions with Firebase and JWT cookies.
This solution is refactored from the ‘streamlit_authenticator’ package . It leverages JSON
Web Token (JWT) cookies to maintain the user’s login state across browser sessions. For the
backend, It uses Google’s Firebase Admin Python SDK. This solution ensures that the content
of the page and user settings panel are only displayed if the user is authenticated. Similarly,
the login page can only be accessed if the user is not authenticated. Upon registration, the
user is sent a verification link to their e-mail address.
Important - to make this app run, put the following variables in your secrets.toml file:
COOKIE_KEY - a random string key for your passwordless reauthentication
FIREBASE_API_KEY - Key for your Firebase API (how to find it -
https://firebase.google.com/docs/projects/api-keys#find-api-keys
)
firebase_auth_token - Information extracted from Firebase login token JSON (how to get one -
https://firebase.google.com/docs/admin/setup#initialize_the_sdk_in_non-google_environments
)
"""

import os
import logging
import json
import math
import time
from contextlib import suppress
from datetime import datetime, timedelta
from functools import partial
from typing import Dict, Final, Optional, Sequence, Union

import extra_streamlit_components as stx
import firebase_admin
import jwt
import requests
import streamlit as st
from email_validator import EmailNotValidError, validate_email
from firebase_admin import auth
from st_pages import Page, show_pages, hide_pages
from firebase_admin import firestore
from PIL import Image
from utils import get_user_data, get_user_collection, display_record, eval_password, encode_payload, get_time
from models import Tags0

TITLE: Final = "REFLEXIVE.AI"
VERSION = "0.0.1"

# Create and configure logger
logging.basicConfig(
    filename="app.log",
    format='%(asctime)s %(message)s',
    filemode='w', )

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


POST_REQUEST_URL_BASE: Final = "https://identitytoolkit.googleapis.com/v1/accounts:"
post_request = partial(
    requests.post,
    headers={"content-type": "application/json; charset=UTF-8"},
    timeout=10,
)
success = partial(st.success, icon="✅")
error = partial(st.error, icon="🚨")


def pretty_title(title: str) -> None:
    """Make a centered title, and give it a red line. Adapted from
    'streamlit_extras.colored_headers' package.
    Parameters:
    -----------
    title : str
        The title of your page.
    """
    st.markdown(
        f"<h2 style='text-align: center'>{title}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<hr style="background-color: #ff4b4b; margin-top: 0;'
            ' margin-bottom: 0; height: 3px; border: none; border-radius: 3px;">'
        ),
        unsafe_allow_html=True,
    )


def parse_error_message(response: requests.Response) -> str:
    """
    Parses an error message from a requests.Response object and makes it look better.
    Parameters:
        response (requests.Response): The response object to parse.
    Returns:
        str: Prettified error message.
    Raises:
        KeyError: If the 'error' key is not present in the response JSON.
    """
    return (
        response.json()["error"]["message"]
        .casefold()
        .replace("_", " ")
        .replace("email", "e-mail")
    )


def authenticate_user(
        email: str, password: str, require_email_verification: bool = True
) -> Optional[Dict[str, Union[str, bool, int]]]:
    """
    Authenticates a user with the given email and password using the Firebase Authentication
    REST API.
    Parameters:
        email (str): The email address of the user to authenticate.
        password (str): The password of the user to authenticate.
        require_email_verification (bool): Specify whether a user has to be e-mail verified to
        be authenticated
    Returns:
        dict or None: A dictionary containing the authenticated user's ID token, refresh token,
        and other information, if authentication was successful. Otherwise, None.
    Raises:
        requests.exceptions.RequestException: If there was an error while authenticating the user.
    """

    url = f"{POST_REQUEST_URL_BASE}signInWithPassword?key={st.secrets['FIREBASE_API_KEY']}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
        "emailVerified": require_email_verification,
    }
    response = post_request(url, json=payload)
    if response.status_code != 200:
        error(f"Authentication failed: {parse_error_message(response)}")
        return None
    response = response.json()
    if require_email_verification and "idToken" not in response:
        error("Invalid e-mail or password.")
        return None
    return response


def forgot_password_form(preauthorized: Union[str, Sequence[str], None]) -> None:
    """Creates a Streamlit widget to reset a user's password. Authentication uses
    the Firebase Authentication REST API.
    Parameters:
        preauthorized (Union[str, Sequence[str], None]): An optional domain or a list of
        domains which are authorized to register.
    """

    with st.form("Forgot password"):
        email = st.text_input("E-mail", key="forgot_password")
        if not st.form_submit_button("Reset password"):
            return None
    if "@" not in email and isinstance(preauthorized, str):
        email = f"{email}@{preauthorized}"

    url = f"{POST_REQUEST_URL_BASE}sendOobCode?key={st.secrets['FIREBASE_API_KEY']}"
    payload = {"requestType": "PASSWORD_RESET", "email": email}
    response = post_request(url, json=payload)
    if response.status_code == 200:
        return success(f"Password reset link has been sent to {email}")
    return error(f"Error sending password reset email: {parse_error_message(response)}")


def register_user_form(preauthorized: Union[str, Sequence[str], None]) -> None:
    """Creates a Streamlit widget for user registration.
    Password strength is validated using entropy bits (the power of the password alphabet).
    Upon registration, a validation link is sent to the user's email address.
    Parameters:
        preauthorized (Union[str, Sequence[str], None]): An optional domain or a list of
        domains which are authorized to register.
    """

    with st.form(key="register_form"):
        email, name, password, confirm_password, register_button = (
            st.text_input("E-mail"),
            st.text_input("Name"),
            st.text_input("Password", type="password"),
            st.text_input("Confirm password", type="password"),
            st.form_submit_button(label="Submit"),
        )
        st.markdown("Password guidelines\n1. at least 12 characters\n2. at least 2 uppercase characters\n3. at least "
                    "2 numbers\n4. at least 2 special characters")
    if not register_button:
        return None
    # Below are some checks to ensure proper and secure registration
    if password != confirm_password:
        return error("Passwords do not match")
    if not name:
        return error("Please enter your name")
    if "@" not in email and isinstance(preauthorized, str):
        email = f"{email}@{preauthorized}"
    if preauthorized and not email.endswith(preauthorized):
        return error("Domain not allowed")
    try:
        validate_email(email, check_deliverability=True)
    except EmailNotValidError as e:
        return error(e)

    password_eval, password_criterias = eval_password(password)
    if not password_eval:
        return st.warning(
            f"Password is too weak: {password_criterias} ", icon="⚠️"
        )
    auth.create_user(
        email=email, password=password, display_name=name, email_verified=False
    )
    # Having registered the user, send them a verification e-mail
    token = authenticate_user(email, password, require_email_verification=False)[
        "idToken"
    ]
    url = f"{POST_REQUEST_URL_BASE}sendOobCode?key={st.secrets['FIREBASE_API_KEY']}"
    payload = {"requestType": "VERIFY_EMAIL", "idToken": token}
    response = post_request(url, json=payload)
    if response.status_code != 200:
        return error(f"Error sending verification email: {parse_error_message(response)}")
    return st.success(
        "Your account has been created successfully. To complete the registration process, "
        "please verify your email address by clicking on the link we have sent to your inbox."
    )


def update_password_form() -> None:
    """Creates a Streamlit widget to update a user's password."""

    # Get the email and password from the user
    new_password = st.text_input("New password", key="new_password")
    # Attempt to log the user in
    if not st.button("Update password", help="update your password"):
        return None
    user = auth.get_user_by_email(st.session_state["username"])
    auth.update_user(user.uid, password=new_password)
    return success("Successfully updated user password.")


def update_display_name_form(
        cookie_manager: stx.CookieManager, cookie_name: str, cookie_expiry_days: int = 30
) -> None:
    """Creates a Streamlit widget to update a user's display name.
    Parameters
    ----------
     - cookie_manager : stx.CookieManager
        A JWT cookie manager instance for Streamlit
    - cookie_name : str
        The name of the reauthentication cookie.
    - cookie_expiry_days: (optional) str
        An integer representing the number of days until the cookie expires
    """

    # Get the email and password from the user
    new_name = st.text_input("New name", key="new name")
    # Attempt to log the user in
    if not st.button("Update name"):
        return None
    user = auth.get_user_by_email(st.session_state["username"])
    auth.update_user(user.uid, display_name=new_name)
    st.session_state["name"] = new_name
    # Update the cookie as well
    exp_date = datetime.utcnow() + timedelta(days=cookie_expiry_days)
    cookie_manager.set(
        cookie_name,
        token_encode(exp_date),
        expires_at=exp_date,
    )
    return success("Successfully updated user display name.")


def token_encode(exp_date: datetime) -> str:
    """Encodes a JSON Web Token (JWT) containing user session data for passwordless
    reauthentication.
    Parameters
    ----------
    exp_date : datetime
        The expiration date of the JWT.
    Returns
    -------
    str
        The encoded JWT cookie string for reauthentication.
    Notes
    -----
    The JWT contains the user's name, username, and the expiration date of the JWT in
    timestamp format. The `st.secrets["COOKIE_KEY"]` value is used to sign the JWT with
    the HS256 algorithm.
    """
    return jwt.encode(
        {
            "name": st.session_state["name"],
            "username": st.session_state["username"],
            "exp_date": exp_date.timestamp(),
        },
        st.secrets["COOKIE_KEY"],
        algorithm="HS256",
    )


def cookie_is_valid(cookie_manager: stx.CookieManager, cookie_name: str) -> bool:
    """Check if the reauthentication cookie is valid and, if it is, update the session state.
    Parameters
    ----------
     - cookie_manager : stx.CookieManager
        A JWT cookie manager instance for Streamlit
    - cookie_name : str
        The name of the reauthentication cookie.
    - cookie_expiry_days: (optional) str
        An integer representing the number of days until the cookie expires
    Returns
    -------
    bool
        True if the cookie is valid and the session state is updated successfully; False otherwise.
    Notes
    -----
    This function checks if the specified reauthentication cookie is present in the cookies stored by
    the cookie manager, and if it is valid. If the cookie is valid, this function updates the session
    state of the Streamlit app and authenticates the user.
    """

    token = cookie_manager.get(cookie_name)
    if token is None:
        return False
    with suppress(Exception):
        token = jwt.decode(token, st.secrets["COOKIE_KEY"], algorithms=["HS256"])
    if (
            token
            and not st.session_state["logout"]
            and token["exp_date"] > datetime.utcnow().timestamp()
            and {"name", "username"}.issubset(set(token))
    ):
        st.session_state["name"] = token["name"]
        st.session_state["username"] = token["username"]
        st.session_state["authentication_status"] = True
        st.session_state["login_connection_time"] = get_time()
        return True
    return False


def login_form(
        cookie_manager: stx.CookieManager,
        cookie_name: str,
        preauthorized: Union[str, Sequence[str], None],
        cookie_expiry_days: int = 30,
) -> None:
    """Creates a login widget using Firebase REST API and a cookie manager.
    Parameters
    ----------
     - cookie_manager : stx.CookieManager
        A JWT cookie manager instance for Streamlit
    - cookie_name : str
        The name of the reauthentication cookie.
    - cookie_expiry_days: (optional) str
        An integer representing the number of days until the cookie expires
    Notes
    -----
    If the user has already been authenticated, this function does nothing. Otherwise, it displays
    a login form which prompts the user to enter their email and password. If the login credentials
    are valid and the user's email address has been verified, the user is authenticated and a
    reauthentication cookie is created with the specified expiration date.
    """

    if st.session_state["authentication_status"]:
        return None
    with st.form("Login"):
        email = st.text_input("E-mail")
        if "@" not in email and isinstance(preauthorized, str):
            email = f"{email}@{preauthorized}"
        st.session_state["username"] = email
        password = st.text_input("Password", type="password")
        if not st.form_submit_button("Login"):
            return None

    # Authenticate the user with Firebase REST API
    login_response = authenticate_user(email, password)
    # print(f"in login_form, login_response:{login_response}")
    if not login_response:
        return None
    try:
        decoded_token = auth.verify_id_token(login_response["idToken"])
        # print(f"in login_form, decoded_token:{decoded_token}")
        user = auth.get_user(decoded_token["uid"])
        if not user.email_verified:
            return error("Please verify your e-mail address.")
        # At last, authenticate the user
        # print(f"in login_form, user:{login_response}")
        st.session_state["name"] = user.display_name
        st.session_state["username"] = user.email
        st.session_state["authentication_status"] = True
        st.session_state["login_connection_time"] = datetime.now()
        exp_date = datetime.utcnow() + timedelta(days=cookie_expiry_days)
        cookie_manager.set(
            cookie_name,
            token_encode(exp_date),
            expires_at=exp_date,
        )
    except Exception as e:
        error(e)
    return None


def login_panel(
        cookie_manager: stx.CookieManager, cookie_name: str, cookie_expiry_days: int = 30
) -> None:
    """Creates a side panel for logged-in users, preventing the login menu from
    appearing.
    Parameters
    ----------
     - cookie_manager : stx.CookieManager
        A JWT cookie manager instance for Streamlit
    - cookie_name : str
        The name of the reauthentication cookie.
    - cookie_expiry_days: (optional) str
        An integer representing the number of days until the cookie expires
    Notes
    -----
    If the user is logged in, this function displays two tabs for resetting the user's password
    and updating their display name.
    If the user clicks the "Logout" button, the reauthentication cookie and user-related information
    from the session state is deleted, and the user is logged out.
    """
    if st.button("Logout", help="quit session"):
        # print(f"in logout from home:{st.session_state}")
        db = firestore.client()  # log in table
        collection_name = st.secrets["firestore_collection"]
        payload = encode_payload(st.session_state)
        doc_ref = db.collection(f"{collection_name}").document()  # create a new document.ID
        doc_ref.set(payload)  # add obj to collection
        db.close()

        cookie_manager.delete(cookie_name)
        st.session_state["logout"] = True
        st.session_state["name"] = None
        st.session_state["username"] = None
        st.session_state["authentication_status"] = None
        st.session_state["login_connection_time"] = None
        st.session_state["comments"] = []

        # clean log messages
        for current_log in ["chat_messages.txt", "qa_messages.txt", "chat_messages.pdf", "qa_messages.pdf"]:
            if os.path.exists(current_log):
                os.remove(current_log)
        return None
    st.write(f"Welcome, *{st.session_state['name']}*!")
    user_tab1, user_tab2 = st.tabs(["Reset password", "Update user details"])
    with user_tab1:
        update_password_form()
    with user_tab2:
        update_display_name_form(cookie_manager, cookie_name, cookie_expiry_days)
    return None


def not_logged_in(
        cookie_manager, cookie_name, preauthorized: Union[str, Sequence[str], None] = None
) -> bool:
    """Creates a tab panel for unauthenticated, preventing the user control sidebar and
    the rest of the script from appearing until the user logs in.
    Parameters
    ----------
     - cookie_manager : stx.CookieManager
        A JWT cookie manager instance for Streamlit
    - cookie_name : str
        The name of the reauthentication cookie.
    - cookie_expiry_days: (optional) str
        An integer representing the number of days until the cookie expires
    Returns
    -------
    Authentication status boolean.
    Notes
    -----
    If the user is already authenticated, the login panel function is called to create a side
    panel for logged-in users. If the function call does not update the authentication status
    because the username/password does not exist in the Firebase database, the rest of the script
    does not get executed until the user logs in.
    """

    early_return = True
    # In case of a first run, pre-populate missing session state arguments
    for key in {"name", "authentication_status", "username", "logout", "login_connection_time",
                "messages_QA", "messages_chat", "userdata", "life_insurance_model"}.difference(
            set(st.session_state)
    ):
        if key in ["messages_chat", "messages_QA"]:
            st.session_state[key] = []
        elif key == 'life_insurance_model':
            user_details = Tags0()
            st.session_state[key] = user_details
        else:
            st.session_state[key] = None

    login_tabs = st.empty()
    with login_tabs:
        login_tab1, login_tab2, login_tab3 = st.tabs(
            ["Login", "Register", "Forgot password"]
        )
        with login_tab1:
            login_form(cookie_manager, cookie_name, preauthorized)
        with login_tab2:
            register_user_form(preauthorized)
        with login_tab3:
            forgot_password_form(preauthorized)

    auth_status = st.session_state["authentication_status"]
    if auth_status is False:
        error("Username/password is incorrect")
        return early_return
    if auth_status is None:
        return early_return
    login_tabs.empty()
    # A workaround for a bug in Streamlit -
    # https://playground.streamlit.app/?q=empty-doesnt-work
    # TLDR: element.empty() doesn't actually seem to work with a multi-element container
    # unless you add a sleep after it.
    time.sleep(0.01)
    return not early_return


def app() -> None:
    """This is a part of a Streamlit app which is only visible if the user is logged in."""
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    st.subheader('Home page')
    st.write(":green[You are logged in!]")
    st.write("Here you can see your current settings")
    # print(st.session_state)
    # retrieve user's data
    # disable for now
    # if st.session_state['userdata'] is None:
    #     user_df = get_user_data(st.session_state['username'])
    #     st.session_state['userdata'] = user_df
    # else:
    #     user_df = st.session_state['userdata']
    # if not user_df.empty:
    #     tmp_df = st.session_state['userdata']
    #     st.table(tmp_df)

    max_records = 5
    user_dates, user_records = get_user_collection(st.secrets["firestore_collection"],
                                                   st.session_state['username'],
                                                   max_records)
    if len(user_dates) > 0:
        # this user has at least 1 record to show
        with st.form("display_record"):
            record = st.selectbox(f':blue[Select record to display(last {max_records} records shown)]',
                                  tuple(user_dates))
            submitted = st.form_submit_button("Submit")
            if submitted:
                display_record(record, user_records)
    else:
        st.text("No existing records")

    st.markdown(
        """
        ---
        Created with ❤️ by reflexive.ai

        """
    )


def main() -> None:
    """Launches a Streamlit example interface.
    The interface supports authentication through Firebase REST API and JSON Web Token (JWT)
    cookies. To use the portal, the user must be registered, optionally only with a preauthorized
    e-mail domain. The Firebase REST API and JWT cookies are used for authentication. If the user
    is not logged in, no content other than the login form gets shown.
    """
    img = Image.open("reflexive_ai_logo.png")
    st.set_page_config(page_title="Reflexive.ai",
                       page_icon=img,
                       initial_sidebar_state="collapsed",
                       menu_items={
                           'Get Help': "mailto:support@reflexive.ai",
                           'Report a bug': "mailto:support@reflexive.ai",
                           'About': "# REFLEXIVE.AI\n ## VIRTUAL AGENTS TO HELP YOU GROW YOUR BUSINESS "
                                    "\nhttps://reflexive.ai"})
    hide_pages([Page("simple_chat.py", "Virtual Insurance Agent"),
                Page("qa_docs.py", "QA Documents")])

    # noinspection PyProtectedMember
    if not firebase_admin._apps:
        # print(f"firebase_admin._apps initialization")
        data = {
            "type": st.secrets['type'],
            "project_id": st.secrets['project_id'],
            "private_key_id": st.secrets['private_key_id'],
            "private_key": st.secrets['private_key'],
            "client_email": st.secrets['client_email'],
            "client_id": st.secrets['client_id'],
            "auth_uri": st.secrets['auth_uri'],
            "token_uri": st.secrets['token_uri'],
            "auth_provider_x509_cert_url": st.secrets['auth_provider_x509_cert_url'],
            "client_x509_cert_url": st.secrets['client_x509_cert_url'],
            "universe_domain": st.secrets['universe_domain']}

        cred = firebase_admin.credentials.Certificate(data)

        firebase_admin.initialize_app(cred)
    # print(f"firebase_admin._apps already initialized")
    st.session_state['first_login'] = True
    # print(f"after firebase admin:{st.session_state}")
    pretty_title(TITLE)
    cookie_manager, cookie_name = stx.CookieManager(), "login_cookie"

    if not cookie_is_valid(cookie_manager, cookie_name) and not_logged_in(
                cookie_manager, cookie_name, preauthorized=None
    ):
        return None

    with st.sidebar:
        st.markdown(f"""version {VERSION}""")
        login_panel(cookie_manager, cookie_name)
        show_pages([
            Page("app.py", "Home", "🏠"),
            Page("simple_chat.py", "Virtual Insurance Agent", "🤖"),
            Page("qa_docs.py", "QA Docs", "❓")
        ])
    return app()


# Run the Streamlit app
if __name__ == "__main__":
    # print(f"Files and directories in {os.getcwd()}: {os.listdir(os.getcwd())}")
    for name in ["chat_messages.txt", "qa_messages.txt", "chat_messages.pdf", "qa_messages.pdf"]:
        if os.path.exists(name):
            os.remove(name)
    main()
