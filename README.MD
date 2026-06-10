
# Python API documentation for https://335456050a3feecb59.gradio.live/
API Endpoints: 3

1. Install the Python client [docs](https://www.gradio.app/guides/getting-started-with-the-python-client) if you don't already have it installed. 

```bash
pip install gradio_client
```

2. Find the API endpoint below corresponding to your desired function in the app. Copy the code snippet, replacing the placeholder values with your own input data. 

### API Name: /toggle_custom


```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
result = client.predict(
	choice="💰 Budget Shopper",
	api_name="/toggle_custom",
)
print(result)
```

Accepts 1 parameter:

choice:
- Type: Literal['🚗 Truck Driver', '💰 Budget Shopper', '👑 Premium Buyer', '🏃 Sports Enthusiast', '🥗 Health Conscious', '🎁 Gift Buyer', '👨‍👩‍👧‍👦 Family Buyer', '🔧 Tech Geek', '📚 Student', 'Custom']
- Default: "💰 Budget Shopper"
- The input value that is provided in the Preset Personas Dropdown component. 

Returns 1 element:





### API Name: /generate_recommendations
Description: Generate recommendations for a selected or custom persona.

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
result = client.predict(
	persona_preset="💰 Budget Shopper",
	custom_persona_name="Hello!!",
	custom_signals="Hello!!",
	custom_query="Hello!!",
	top_k=15,
	api_name="/generate_recommendations",
)
print(result)
```

Accepts 5 parameters:

persona_preset:
- Type: Literal['🚗 Truck Driver', '💰 Budget Shopper', '👑 Premium Buyer', '🏃 Sports Enthusiast', '🥗 Health Conscious', '🎁 Gift Buyer', '👨‍👩‍👧‍👦 Family Buyer', '🔧 Tech Geek', '📚 Student', 'Custom']
- Default: "💰 Budget Shopper"
- The input value that is provided in the Preset Personas Dropdown component. 

custom_persona_name:
- Type: str
- Required
- The input value that is provided in the Persona Name Textbox component. 

custom_signals:
- Type: str
- Required
- The input value that is provided in the Signals (comma-separated) Textbox component. 

custom_query:
- Type: str
- Required
- The input value that is provided in the Query String Textbox component. 

top_k:
- Type: float
- Default: 15
- The input value that is provided in the Top-K Results Slider component. 

Returns tuple of 2 elements:

[0]: - Type: str
- The output value that appears in the "value_18" Html component.

[1]: - Type: str
- The output value that appears in the "Status" Textbox component.



### API Name: /export_recommendations
Description: Export recommendations as JSON.

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
result = client.predict(
	persona_preset="💰 Budget Shopper",
	custom_persona_name="Hello!!",
	custom_signals="Hello!!",
	custom_query="Hello!!",
	top_k=15,
	api_name="/export_recommendations",
)
print(result)
```

Accepts 5 parameters:

persona_preset:
- Type: Literal['🚗 Truck Driver', '💰 Budget Shopper', '👑 Premium Buyer', '🏃 Sports Enthusiast', '🥗 Health Conscious', '🎁 Gift Buyer', '👨‍👩‍👧‍👦 Family Buyer', '🔧 Tech Geek', '📚 Student', 'Custom']
- Default: "💰 Budget Shopper"
- The input value that is provided in the Preset Personas Dropdown component. 

custom_persona_name:
- Type: str
- Required
- The input value that is provided in the Persona Name Textbox component. 

custom_signals:
- Type: str
- Required
- The input value that is provided in the Signals (comma-separated) Textbox component. 

custom_query:
- Type: str
- Required
- The input value that is provided in the Query String Textbox component. 

top_k:
- Type: float
- Default: 15
- The input value that is provided in the Top-K Results Slider component. 

Returns 1 element:

- Type: str
- The output value that appears in the "Export Status" Textbox component.

