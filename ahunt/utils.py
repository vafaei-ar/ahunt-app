
import streamlit as st

# try:
#     from streamlit.scriptrunner import get_script_run_ctx
# except ModuleNotFoundError:
#     # streamlit < 1.8
#     try:
#         from streamlit.script_run_context import get_script_run_ctx  # type: ignore
#     except ModuleNotFoundError:
#         # streamlit < 1.4
#         from streamlit.report_thread import (  # type: ignore
#             get_report_ctx as get_script_run_ctx,
#         )

# def get_session_id() -> str:
#     ctx = get_script_run_ctx()
#     if not ctx:
#         raise Exception("Failed to get the thread context")

#     return ctx.session_id

# class SessionState(object):
#     def __init__(self, **kwargs):
#         """A new SessionState object.

#         Parameters
#         ----------
#         **kwargs : any
#             Default values for the session state.

#         Example
#         -------
#         >>> session_state = SessionState(user_name='', favorite_color='black')
#         >>> session_state.user_name = 'Mary'
#         ''
#         >>> session_state.favorite_color
#         'black'

#         """
#         for key, val in kwargs.items():
#             setattr(self, key, val)


# @st.cache(allow_output_mutation=True)
# def get_session(id, **kwargs):
#     return SessionState(**kwargs)


# def get(**kwargs):
#     """Gets a SessionState object for the current session.

#     Creates a new object if necessary.

#     Parameters
#     ----------
#     **kwargs : any
#         Default values you want to add to the session state, if we're creating a
#         new one.

#     Example
#     -------
#     >>> session_state = get(user_name='', favorite_color='black')
#     >>> session_state.user_name
#     ''
#     >>> session_state.user_name = 'Mary'
#     >>> session_state.favorite_color
#     'black'

#     Since you set user_name above, next time your script runs this will be the
#     result:
#     >>> session_state = get(user_name='', favorite_color='black')
#     >>> session_state.user_name
#     'Mary'

#     """
#     # ctx = get_report_ctx()
#     # id = ctx.session_id
#     id = get_session_id()
#     return get_session(id, **kwargs)
