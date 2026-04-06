"""Authentication pages: login, register, change password."""
import streamlit as st

from auth_db import authenticate_user, change_password, register_user

_AUTH_HEADER = """
<div style="
  text-align: center;
  padding: 3rem 0 1.5rem;
">
  <div style="
    display: inline-block;
    width: 48px; height: 48px;
    background: #c96442;
    border-radius: 50%;
    margin-bottom: 1rem;
    display: flex; align-items: center; justify-content: center;
  ">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
         xmlns="http://www.w3.org/2000/svg" style="margin:auto;display:block;margin-top:12px;">
      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
            stroke="#faf9f5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
  <div style="
    font-family: 'Lora', Georgia, serif;
    font-weight: 500;
    font-size: 2rem;
    line-height: 1.15;
    color: #141413;
    margin-bottom: 0.4rem;
  ">A/B Test Platform</div>
  <div style="
    font-family: Inter, system-ui, sans-serif;
    font-size: 0.94rem;
    color: #87867f;
    line-height: 1.60;
  ">Statistical analysis for data-driven decisions</div>
</div>
"""


def _tab_button(label: str, active: bool, key: str, tab_name: str) -> None:
    btn_type = "primary" if active else "secondary"
    if st.button(label, width='stretch', type=btn_type, key=key):
        st.session_state["auth_tab"] = tab_name
        st.rerun()


def show_auth_page() -> None:
    st.markdown(_AUTH_HEADER, unsafe_allow_html=True)

    _, col_center, _ = st.columns([1, 1.1, 1])
    with col_center:
        if "auth_tab" not in st.session_state:
            st.session_state["auth_tab"] = "login"

        # Tab switcher
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            _tab_button("Login", st.session_state["auth_tab"] == "login", "btn_login", "login")
        with col_t2:
            _tab_button("Register", st.session_state["auth_tab"] == "register", "btn_register", "register")

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        if st.session_state["auth_tab"] == "login":
            with st.form("login_form"):
                st.markdown(
                    '<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:500;'
                    'letter-spacing:0.08em;text-transform:uppercase;color:#87867f;margin-bottom:0.5rem;">'
                    'Username</p>',
                    unsafe_allow_html=True,
                )
                username = st.text_input("Username", label_visibility="collapsed")
                st.markdown(
                    '<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:500;'
                    'letter-spacing:0.08em;text-transform:uppercase;color:#87867f;'
                    'margin-top:0.75rem;margin-bottom:0.5rem;">Password</p>',
                    unsafe_allow_html=True,
                )
                password = st.text_input("Password", type="password", label_visibility="collapsed")
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Sign In", width='stretch')

            if submitted:
                if not username or not password:
                    st.error("Please enter your username and password.")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state["user"] = user
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        else:
            with st.form("register_form"):
                st.markdown(
                    '<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:500;'
                    'letter-spacing:0.08em;text-transform:uppercase;color:#87867f;margin-bottom:0.5rem;">'
                    'Username</p>',
                    unsafe_allow_html=True,
                )
                new_username = st.text_input("Username", label_visibility="collapsed")
                st.markdown(
                    '<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:500;'
                    'letter-spacing:0.08em;text-transform:uppercase;color:#87867f;'
                    'margin-top:0.75rem;margin-bottom:0.5rem;">Password <span style="color:#b0aea5;'
                    'font-weight:400;">(min. 6 characters)</span></p>',
                    unsafe_allow_html=True,
                )
                new_password = st.text_input("Password", type="password", label_visibility="collapsed")
                st.markdown(
                    '<p style="font-family:Inter,sans-serif;font-size:0.75rem;font-weight:500;'
                    'letter-spacing:0.08em;text-transform:uppercase;color:#87867f;'
                    'margin-top:0.75rem;margin-bottom:0.5rem;">Confirm Password</p>',
                    unsafe_allow_html=True,
                )
                confirm_pwd = st.text_input("Confirm Password", type="password", label_visibility="collapsed")
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                reg_submitted = st.form_submit_button("Create Account", width='stretch')

            if reg_submitted:
                if new_password != confirm_pwd:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(new_username, new_password)
                    if ok:
                        st.session_state["auth_tab"] = "login"
                        st.session_state["register_success"] = msg
                        st.rerun()
                    else:
                        st.error(msg)

        if st.session_state.pop("register_success", None):
            st.success("Account created — please sign in.")

        # Footer note
        st.markdown(
            '<p style="text-align:center;font-size:0.78rem;color:#b0aea5;margin-top:1.25rem;'
            'font-family:Inter,sans-serif;">Secure · Private · Your data stays local</p>',
            unsafe_allow_html=True,
        )


def show_change_password_page() -> None:
    if st.button("← Back to Analysis"):
        st.session_state["page"] = "Analysis"
        st.rerun()

    st.markdown(
        '<h2 style="font-family:\'Lora\',Georgia,serif;font-weight:500;'
        'color:#141413;margin-bottom:0.25rem;">Change Password</h2>',
        unsafe_allow_html=True,
    )
    user = st.session_state["user"]

    col, _ = st.columns([1, 2])
    with col:
        with st.form("change_pwd_form"):
            old_pwd  = st.text_input("Current Password", type="password")
            new_pwd  = st.text_input("New Password (min. 6 characters)", type="password")
            new_pwd2 = st.text_input("Confirm New Password", type="password")
            submitted = st.form_submit_button("Update Password")

        if submitted:
            if new_pwd != new_pwd2:
                st.error("New passwords do not match.")
            else:
                ok, msg = change_password(user["id"], old_pwd, new_pwd)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
