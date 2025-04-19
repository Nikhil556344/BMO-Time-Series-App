{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOt9kPLrleo+NDTuZdkhwAJ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Nikhil556344/BMO-Time-Series-App/blob/main/Streamlitapp_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install streamlit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aMu4jpIvcW3n",
        "outputId": "ca97e1dc-a5a5-40c6-87e3-269acdea5a77"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.44.1-py3-none-any.whl.metadata (8.9 kB)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (8.1.8)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.0.2)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (11.1.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.29.4)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (9.1.2)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.11/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (4.13.2)\n",
            "Collecting watchdog<7,>=2.1.5 (from streamlit)\n",
            "  Downloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl.metadata (44 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m44.3/44.3 kB\u001b[0m \u001b[31m2.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.11/dist-packages (from streamlit) (3.1.44)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.11/dist-packages (from streamlit) (6.4.2)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (1.35.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.11/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.4.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2.3.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2025.1.31)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.11/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (25.3.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2024.10.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.36.2)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.24.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n",
            "Downloading streamlit-1.44.1-py3-none-any.whl (9.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m9.8/9.8 MB\u001b[0m \u001b[31m30.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m35.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl (79 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m79.1/79.1 kB\u001b[0m \u001b[31m5.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: watchdog, pydeck, streamlit\n",
            "Successfully installed pydeck-0.9.1 streamlit-1.44.1 watchdog-6.0.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "C8NiAbbpcIGC",
        "outputId": "5f53a503-5a60-46a7-8ac7-91ac068d0767"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-04-19 23:46:12.961 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-19 23:46:13.651 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2025-04-19 23:46:13.655 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-19 23:46:13.659 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-19 23:46:13.662 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-19 23:46:13.665 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-19 23:46:13.669 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-19 23:46:13.673 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-19 23:46:13.677 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-19 23:46:13.680 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ],
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from statsmodels.tsa.seasonal import seasonal_decompose\n",
        "from statsmodels.tsa.arima.model import ARIMA\n",
        "from statsmodels.tsa.holtwinters import SimpleExpSmoothing\n",
        "from prophet import Prophet\n",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
        "import numpy as np\n",
        "\n",
        "st.title(\" Time Series Forecasting App\")\n",
        "st.markdown(\"Upload your CSV, choose decomposition type and forecasting model to view results.\")\n",
        "\n",
        "# Upload file\n",
        "uploaded_file = st.file_uploader(\"Upload a CSV file\", type=\"csv\")\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    data = pd.read_csv(uploaded_file)\n",
        "    data.columns = data.columns.str.replace('\\xa0', '', regex=True).str.strip()\n",
        "    if 'Date' not in data.columns or 'Close' not in data.columns:\n",
        "        st.error(\"Your file must contain 'Date' and 'Close' columns.\")\n",
        "    else:\n",
        "        data['Date'] = pd.to_datetime(data['Date'])\n",
        "        data.set_index('Date', inplace=True)\n",
        "        st.write(\"### Raw Data Preview\")\n",
        "        st.dataframe(data.head())\n",
        "\n",
        "        # Resample monthly\n",
        "        monthly = data['Close'].resample('M').last().to_frame()\n",
        "\n",
        "        # Decomposition Selection\n",
        "        model_type = st.selectbox(\"Select decomposition type\", [\"Additive\", \"Multiplicative\"])\n",
        "        period = st.slider(\"Seasonal Period\", min_value=2, max_value=60, value=30)\n",
        "\n",
        "        try:\n",
        "            decomposition = seasonal_decompose(monthly['Close'].dropna(), model=model_type.lower(), period=period)\n",
        "            st.write(f\"### {model_type} Decomposition\")\n",
        "            fig, axs = plt.subplots(4, 1, figsize=(12, 10))\n",
        "            axs[0].plot(decomposition.observed); axs[0].set_title(\"Original\")\n",
        "            axs[1].plot(decomposition.trend); axs[1].set_title(\"Trend\")\n",
        "            axs[2].plot(decomposition.seasonal); axs[2].set_title(\"Seasonal\")\n",
        "            axs[3].plot(decomposition.resid); axs[3].set_title(\"Residual\")\n",
        "            plt.tight_layout()\n",
        "            st.pyplot(fig)\n",
        "        except Exception as e:\n",
        "            st.error(f\"Decomposition Error: {e}\")\n",
        "\n",
        "        # Forecasting Model\n",
        "        model_choice = st.selectbox(\"Choose a forecasting model\", [\"ARIMA\", \"ETS\", \"Prophet\"])\n",
        "        split = int(len(monthly) * 0.8)\n",
        "        train, test = monthly[:split], monthly[split:]\n",
        "        forecast_index = test.index\n",
        "\n",
        "        def evaluate(true, pred):\n",
        "            mse = mean_squared_error(true, pred)\n",
        "            rmse = np.sqrt(mse)\n",
        "            mae = mean_absolute_error(true, pred)\n",
        "            mape = np.mean(np.abs((true - pred) / true)) * 100\n",
        "            return rmse, mae, mape, mse\n",
        "\n",
        "        if model_choice == \"ARIMA\":\n",
        "            try:\n",
        "                model_arima = ARIMA(train, order=(5, 1, 5)).fit()\n",
        "                forecast = model_arima.predict(start=len(train), end=len(train) + len(test) - 1)\n",
        "                forecast.index = forecast_index\n",
        "                st.line_chart(pd.DataFrame({'Actual': test['Close'], 'Forecast': forecast}))\n",
        "                rmse, mae, mape, mse = evaluate(test['Close'], forecast)\n",
        "            except Exception as e:\n",
        "                st.error(f\"ARIMA Error: {e}\")\n",
        "\n",
        "        elif model_choice == \"ETS\":\n",
        "            try:\n",
        "                model_ets = SimpleExpSmoothing(monthly['Close']).fit(smoothing_level=0.2)\n",
        "                forecast = model_ets.predict(start=test.index[0], end=test.index[-1])\n",
        "                st.line_chart(pd.DataFrame({'Actual': test['Close'], 'Forecast': forecast}))\n",
        "                rmse, mae, mape, mse = evaluate(test['Close'], forecast)\n",
        "            except Exception as e:\n",
        "                st.error(f\"ETS Error: {e}\")\n",
        "\n",
        "        elif model_choice == \"Prophet\":\n",
        "            try:\n",
        "                prophet_data = train.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})\n",
        "                future = test.reset_index().rename(columns={'Date': 'ds'})\n",
        "                m = Prophet(yearly_seasonality=True)\n",
        "                m.fit(prophet_data)\n",
        "                future_df = m.make_future_dataframe(periods=len(test), freq='M')\n",
        "                forecast_df = m.predict(future_df)\n",
        "                forecast = forecast_df.set_index('ds').loc[forecast_index]['yhat']\n",
        "                st.line_chart(pd.DataFrame({'Actual': test['Close'], 'Forecast': forecast}))\n",
        "                rmse, mae, mape, mse = evaluate(test['Close'], forecast)\n",
        "            except Exception as e:\n",
        "                st.error(f\"Prophet Error: {e}\")\n",
        "\n",
        "        # Show evaluation\n",
        "        st.write(\"###  Forecast Evaluation Metrics\")\n",
        "        st.metric(\"RMSE\", f\"{rmse:.2f}\")\n",
        "        st.metric(\"MAE\", f\"{mae:.2f}\")\n",
        "        st.metric(\"MAPE\", f\"{mape:.2f}%\")\n",
        "        st.metric(\"MSE\", f\"{mse:.2f}\")\n"
      ]
    }
  ]
}