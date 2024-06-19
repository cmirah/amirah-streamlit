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
        page_title="Epidemic Forecast Pro App",
        page_icon=":balloon:",
        layout="centered"
    )

    st.title("👋 Epidemic Forecast Pro App 👩‍💻")

    # Display image
    st.image("WEBCoronavirus.png", width=500)

    # Sidebar and title
    st.sidebar.success("Select an option above.")

    st.markdown(
        """
        Welcome to the **Epidemic Forecast Pro App**. This web-based application predicts the number of COVID-19 cases in the future.
        
        **👈 Select a demo from the sidebar** to explore the features of this app!
        """
    )
    st.image(
            "https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2F%40creatrohit9%2Fmachine-learning-introduction-5049601d2da2&psig=AOvVaw0oNMCd1uzhvpwid3_wAgdU&ust=1718885882997000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCJjUgIbT54YDFQAAAAAdAAAAABAQ",
            width=400, 
        )
    
    st.header("More Information")
    st.markdown(
        """
        - This application was developed by **Amirah Alysha bt Azman** as part of her Final Year Project II.
        - Data sourced from the [MoH-Malaysia](https://github.com/MoH-Malaysia/covid19-public.git).
        - Detailed Python coding for each S-I-R-F Prediction can be found here:
    """
    )

    st.markdown("- [Susceptible](https://colab.research.google.com/drive/1aclFhSdOjodKFvyAKyfxm53u8HhGLGPy?usp=sharing)")
    st.markdown("- [Infected](https://colab.research.google.com/drive/1uVmQawkhqjRUMAzVYZ4pwdc0aV4wbUvG?usp=sharing)")
    st.markdown("- [Recovered](https://colab.research.google.com/drive/1rDhDSnG38JYtLy42Kc0o09gL5jNvuIF9?usp=sharing)")
    st.markdown("- [Fatal](https://colab.research.google.com/drive/1BhEDXYKhtgjf3yj9F-s8_ytwfI15B1e1?usp=sharing)")

if __name__ == "__main__":
    run()

