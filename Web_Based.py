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

    st.write("# 👋 Prediction of Covid-19 Using SIR-F Model App 👩‍💻")

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
        - This Web-Based apps is solely developed by 
        ***Amirah Alysha bt Azman*** in completion of Final Year Project II
        - Data from [MoH-Malaysia](https://github.com/MoH-Malaysia/covid19-public.git)
        - Jump into my [amirah's app](https://share.streamlit.io/)
        - Visit my Python coding for each S-I-R-F Prediction:
        [Susceptible Prediction](https://colab.research.google.com/drive/1aclFhSdOjodKFvyAKyfxm53u8HhGLGPy?usp=sharing)
        [Infected Prediction](https://colab.research.google.com/drive/1uVmQawkhqjRUMAzVYZ4pwdc0aV4wbUvG?usp=sharing)
        [Recovered Prediction](https://colab.research.google.com/drive/1rDhDSnG38JYtLy42Kc0o09gL5jNvuIF9?usp=sharing)
        [Fatal Prediction](https://colab.research.google.com/drive/1BhEDXYKhtgjf3yj9F-s8_ytwfI15B1e1?usp=sharing)
        
        
        
    """
    )


if __name__ == "__main__":
    run()
