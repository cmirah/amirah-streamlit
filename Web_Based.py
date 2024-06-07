# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Web-Based",
        page_icon=":balloon:",
    )

    st.write("# 👋 Welcome to Web-Based! 👩‍💻")

    #image
    st.image("WEBCoronavirus.png",width=500)

    #title
    st.sidebar.success("Select an option above.")

    st.markdown(
        """
        This web-based app is an application used to predict the number of COVID-19 disease that will be occured in the future.
        **👈 Select a demo from the sidebar** to see some examples
        of what this app can do!
        ### More information
        - Data from [MoH-Malaysia](https://github.com/MoH-Malaysia/covid19-public.git)
        - Jump into my [amirah's app](https://share.streamlit.io/)
        - Ask a question in Streamlit's [community
          forums](https://discuss.streamlit.io)
    """
    )


if __name__ == "__main__":
    run()
