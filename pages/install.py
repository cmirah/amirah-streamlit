import streamlit as st
import subprocess

st.title('Update pip within Streamlit')

if st.button('Update pip'):
    # Run the pip install --upgrade pip command
    result = subprocess.run(['pip', 'install', '--upgrade', 'pip'], capture_output=True, text=True)
    if result.returncode == 0:
        st.success('pip has been successfully updated!')
        st.text(result.stdout)
    else:
        st.error('Failed to update pip.')
        st.text(result.stderr)
