# Extending Friday with Integrations

This guide shows how to extend Friday's capabilities by integrating with external services and APIs to create a more powerful assistant.

## Weather Integration

Add weather forecasting capabilities by integrating with a weather API:

### 1. Create a Weather Service Module

Create a new file at `friday/services/weather.py`:

```python
import requests
import json
import asyncio

class WeatherService:
    """Service for fetching weather information"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    async def get_current_weather(self, city):
        """Get current weather for a city"""
        try:
            url = f"{self.base_url}/weather?q={city}&appid={self.api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            if response.status_code != 200:
                return f"Error getting weather: {data.get('message', 'Unknown error')}"
            
            # Extract relevant information
            weather_desc = data['weather'][0]['description']
            temperature = data['main']['temp']
            feels_like = data['main']['feels_like']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            
            weather_info = (
                f"Current weather in {city}:\n"
                f"- Condition: {weather_desc}\n"
                f"- Temperature: {temperature}°C (feels like {feels_like}°C)\n"
                f"- Humidity: {humidity}%\n"
                f"- Wind speed: {wind_speed} m/s"
            )
            
            return weather_info
            
        except Exception as e:
            return f"Error fetching weather data: {str(e)}"
    
    async def get_forecast(self, city, days=3):
        """Get weather forecast for a city"""
        try:
            url = f"{self.base_url}/forecast?q={city}&appid={self.api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            if response.status_code != 200:
                return f"Error getting forecast: {data.get('message', 'Unknown error')}"
            
            # Process forecast data (simplified)
            forecasts = []
            processed_dates = set()
            
            for item in data['list']:
                date = item['dt_txt'].split(' ')[0]
                
                # Only include one forecast per day
                if date in processed_dates:
                    continue
                    
                processed_dates.add(date)
                if len(processed_dates) > days:
                    break
                
                weather_desc = item['weather'][0]['description']
                temp_max = item['main']['temp_max']
                temp_min = item['main']['temp_min']
                
                forecasts.append(
                    f"Date: {date}\n"
                    f"- Condition: {weather_desc}\n"
                    f"- Temperature: {temp_min}°C to {temp_max}°C"
                )
            
            return "Weather Forecast:\n\n" + "\n\n".join(forecasts)
            
        except Exception as e:
            return f"Error fetching forecast data: {str(e)}"
```

### 2. Update Configuration File

Add weather API settings in your `config.yaml`:

```yaml
# Service integrations
services:
  weather:
    enabled: true
    api_key: "your-openweathermap-api-key"
```

### 3. Add Command Handler

Update `main.py` to handle weather commands:

```python
# Inside the main loop where commands are processed
elif user_input.lower().startswith('weather '):
    city = user_input[8:].strip()  # Extract city name after "weather "
    if city:
        print(f"Getting weather for {city}...")
        from friday.services.weather import WeatherService
        
        # Initialize the weather service
        weather_service = WeatherService(config['services']['weather']['api_key'])
        
        # Get weather info
        weather_info = await weather_service.get_current_weather(city)
        print(weather_info)
        tts.generate_speech(weather_info)
    else:
        print("Please specify a city name after 'weather'")

elif user_input.lower().startswith('forecast '):
    city = user_input[9:].strip()  # Extract city name after "forecast "
    if city:
        print(f"Getting forecast for {city}...")
        from friday.services.weather import WeatherService
        
        # Initialize the weather service
        weather_service = WeatherService(config['services']['weather']['api_key'])
        
        # Get forecast info
        forecast_info = await weather_service.get_forecast(city)
        print(forecast_info)
        tts.generate_speech(forecast_info)
    else:
        print("Please specify a city name after 'forecast'")
```

## Smart Home Integration

Connect Friday to smart home devices using Home Assistant:

### 1. Create a Smart Home Service Module

Create a new file at `friday/services/smarthome.py`:

```python
import requests
import json

class HomeAssistantService:
    """Service for controlling smart home devices via Home Assistant"""
    
    def __init__(self, base_url, access_token):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    async def get_devices(self):
        """Get list of available devices"""
        try:
            response = requests.get(f"{self.base_url}/api/states", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting devices: {e}")
            return []
    
    async def turn_on_device(self, entity_id):
        """Turn on a device by entity ID"""
        try:
            data = {"entity_id": entity_id}
            response = requests.post(
                f"{self.base_url}/api/services/switch/turn_on",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            return f"Turned on {entity_id}"
        except Exception as e:
            return f"Error turning on device: {e}"
    
    async def turn_off_device(self, entity_id):
        """Turn off a device by entity ID"""
        try:
            data = {"entity_id": entity_id}
            response = requests.post(
                f"{self.base_url}/api/services/switch/turn_off",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            return f"Turned off {entity_id}"
        except Exception as e:
            return f"Error turning off device: {e}"
    
    async def set_light_brightness(self, entity_id, brightness):
        """Set brightness of a light (0-255)"""
        try:
            data = {
                "entity_id": entity_id,
                "brightness": brightness
            }
            response = requests.post(
                f"{self.base_url}/api/services/light/turn_on",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            return f"Set {entity_id} brightness to {brightness}"
        except Exception as e:
            return f"Error setting brightness: {e}"
```

### 2. Update Configuration File

Add Home Assistant settings in your `config.yaml`:

```yaml
# Service integrations
services:
  # Weather settings...
  
  home_assistant:
    enabled: true
    base_url: "http://192.168.1.100:8123"  # Your Home Assistant URL
    access_token: "your-long-lived-access-token"
    # Define common device mappings to make commands more natural
    device_map:
      "living room light": "light.living_room"
      "kitchen light": "light.kitchen"
      "bedroom light": "light.bedroom"
      "tv": "switch.tv"
      "heater": "climate.living_room"
```

### 3. Add Natural Language Processing for Smart Home Commands

Create a command interpreter for smart home devices at `friday/services/command_parser.py`:

```python
import re

class CommandParser:
    """Parse natural language commands for smart home control"""
    
    def __init__(self, device_map):
        self.device_map = device_map
    
    def parse_command(self, text):
        """
        Parse natural language command text
        Returns: (command_type, device_name, parameter)
        """
        text = text.lower()
        
        # Turn on commands
        turn_on_match = re.search(r'turn on(?: the)? (.+)', text)
        if turn_on_match:
            device_name = turn_on_match.group(1).strip()
            return ('turn_on', device_name, None)
        
        # Turn off commands
        turn_off_match = re.search(r'turn off(?: the)? (.+)', text)
        if turn_off_match:
            device_name = turn_off_match.group(1).strip()
            return ('turn_off', device_name, None)
        
        # Set brightness commands
        brightness_match = re.search(r'set(?: the)? (.+?) (?:brightness|level) to (\d+)(?:\s*%)?', text)
        if brightness_match:
            device_name = brightness_match.group(1).strip()
            brightness = int(brightness_match.group(2))
            # Convert percentage to 0-255 scale
            if brightness <= 100:
                brightness = int(brightness * 2.55)
            return ('set_brightness', device_name, brightness)
        
        # No recognized command
        return (None, None, None)
    
    def get_entity_id(self, device_name):
        """Convert natural device name to entity ID"""
        device_name = device_name.lower()
        if device_name in self.device_map:
            return self.device_map[device_name]
        return None
```

### 4. Add Command Handler

Update `main.py` to handle smart home commands:

```python
# Inside the main loop where commands are processed
elif config['services'].get('home_assistant', {}).get('enabled', False):
    # Check if message could be a smart home command
    from friday.services.command_parser import CommandParser
    from friday.services.smarthome import HomeAssistantService
    
    # Initialize parser and service
    parser = CommandParser(config['services']['home_assistant']['device_map'])
    home_assistant = HomeAssistantService(
        config['services']['home_assistant']['base_url'],
        config['services']['home_assistant']['access_token']
    )
    
    # Parse command
    command_type, device_name, parameter = parser.parse_command(user_input)
    
    if command_type:
        entity_id = parser.get_entity_id(device_name)
        
        if not entity_id:
            response = f"I don't recognize the device '{device_name}'"
        else:
            if command_type == 'turn_on':
                response = await home_assistant.turn_on_device(entity_id)
            elif command_type == 'turn_off':
                response = await home_assistant.turn_off_device(entity_id)
            elif command_type == 'set_brightness':
                response = await home_assistant.set_light_brightness(entity_id, parameter)
        
        print(response)
        tts.generate_speech(response)
        continue  # Skip the regular LLM call
```

## Calendar Integration

Integrate Friday with Google Calendar:

### 1. Install Google Calendar Dependencies

Add the following to your `requirements.txt`:

```
google-api-python-client
google-auth-httplib2
google-auth-oauthlib
```

### 2. Create a Calendar Service Module

Create a new file at `friday/services/calendar_service.py`:

```python
import os
import datetime
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

class GoogleCalendarService:
    """Service for accessing Google Calendar"""
    
    def __init__(self, credentials_path, token_path):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.scopes = ['https://www.googleapis.com/auth/calendar.readonly']
        self.service = self._get_calendar_service()
    
    def _get_calendar_service(self):
        """Set up the Calendar API service"""
        creds = None
        
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # If credentials don't exist or are invalid, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.scopes)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('calendar', 'v3', credentials=creds)
    
    async def get_upcoming_events(self, max_events=5):
        """Get upcoming events from calendar"""
        try:
            now = datetime.datetime.utcnow().isoformat() + 'Z'
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=max_events,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                return "No upcoming events found."
            
            event_list = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                
                # Convert to datetime object
                if 'T' in start:  # Has time component
                    start_dt = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
                    start_str = start_dt.strftime('%A, %B %d at %I:%M %p')
                else:  # All-day event
                    start_dt = datetime.datetime.fromisoformat(start)
                    start_str = start_dt.strftime('%A, %B %d (all day)')
                
                event_list.append(f"- {start_str}: {event['summary']}")
            
            return "Upcoming Events:\n" + "\n".join(event_list)
            
        except Exception as e:
            return f"Error fetching calendar events: {str(e)}"
```

### 3. Update Configuration File

Add Google Calendar settings in your `config.yaml`:

```yaml
# Service integrations
services:
  # Other services...
  
  google_calendar:
    enabled: true
    credentials_path: "credentials.json"  # OAuth credentials file from Google Cloud Console
    token_path: "token.pickle"  # Where to store the access token
```

### 4. Add Command Handler

Update `main.py` to handle calendar commands:

```python
# Inside the main loop where commands are processed
elif user_input.lower() in ['calendar', 'schedule', 'upcoming events']:
    if config['services'].get('google_calendar', {}).get('enabled', False):
        print("Checking your calendar...")
        from friday.services.calendar_service import GoogleCalendarService
        
        # Initialize the calendar service
        calendar_service = GoogleCalendarService(
            config['services']['google_calendar']['credentials_path'],
            config['services']['google_calendar']['token_path']
        )
        
        # Get upcoming events
        events = await calendar_service.get_upcoming_events()
        print(events)
        tts.generate_speech(events)
    else:
        print("Google Calendar integration is not enabled.")
```

## Notes and Reminders Integration

Create a simple notes and reminders system:

### 1. Create a Notes Service Module

Create a new file at `friday/services/notes.py`:

```python
import json
import os
import datetime

class NotesService:
    """Service for managing notes and reminders"""
    
    def __init__(self, notes_file, reminders_file):
        self.notes_file = notes_file
        self.reminders_file = reminders_file
        
        # Create files if they don't exist
        for file_path in [self.notes_file, self.reminders_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)
    
    async def add_note(self, content, category="general"):
        """Add a new note"""
        try:
            # Load existing notes
            with open(self.notes_file, 'r', encoding='utf-8') as f:
                notes = json.load(f)
            
            # Create new note
            new_note = {
                "id": len(notes) + 1,
                "content": content,
                "category": category,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Add and save
            notes.append(new_note)
            with open(self.notes_file, 'w', encoding='utf-8') as f:
                json.dump(notes, f, indent=2)
            
            return f"Note saved: {content}"
            
        except Exception as e:
            return f"Error saving note: {str(e)}"
    
    async def get_notes(self, category=None, limit=5):
        """Get recent notes, optionally filtered by category"""
        try:
            # Load notes
            with open(self.notes_file, 'r', encoding='utf-8') as f:
                notes = json.load(f)
            
            # Filter by category if specified
            if category and category != "all":
                notes = [note for note in notes if note['category'].lower() == category.lower()]
            
            # Sort by timestamp (newest first)
            notes.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Limit the number of notes
            notes = notes[:limit]
            
            if not notes:
                return "No notes found."
            
            # Format notes
            note_list = []
            for note in notes:
                timestamp = datetime.datetime.fromisoformat(note['timestamp']).strftime('%Y-%m-%d %H:%M')
                note_list.append(f"- [{timestamp}] {note['content']}")
            
            return "Your Notes:\n" + "\n".join(note_list)
            
        except Exception as e:
            return f"Error retrieving notes: {str(e)}"
    
    async def add_reminder(self, content, due_date_str):
        """Add a new reminder with due date"""
        try:
            # Parse due date
            try:
                due_date = datetime.datetime.strptime(due_date_str, "%Y-%m-%d %H:%M")
            except ValueError:
                try:
                    due_date = datetime.datetime.strptime(due_date_str, "%Y-%m-%d")
                    # Set default time to noon if not specified
                    due_date = due_date.replace(hour=12, minute=0)
                except ValueError:
                    return f"Invalid date format. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM"
            
            # Load existing reminders
            with open(self.reminders_file, 'r', encoding='utf-8') as f:
                reminders = json.load(f)
            
            # Create new reminder
            new_reminder = {
                "id": len(reminders) + 1,
                "content": content,
                "due_date": due_date.isoformat(),
                "completed": False,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            # Add and save
            reminders.append(new_reminder)
            with open(self.reminders_file, 'w', encoding='utf-8') as f:
                json.dump(reminders, f, indent=2)
            
            # Format due date for output
            due_str = due_date.strftime('%A, %B %d at %I:%M %p')
            return f"Reminder set: {content} (due {due_str})"
            
        except Exception as e:
            return f"Error setting reminder: {str(e)}"
    
    async def get_reminders(self, include_completed=False):
        """Get upcoming reminders"""
        try:
            # Load reminders
            with open(self.reminders_file, 'r', encoding='utf-8') as f:
                reminders = json.load(f)
            
            # Filter out completed reminders if requested
            if not include_completed:
                reminders = [r for r in reminders if not r['completed']]
            
            # Sort by due date
            reminders.sort(key=lambda x: x['due_date'])
            
            if not reminders:
                return "No upcoming reminders."
            
            # Format reminders
            now = datetime.datetime.now()
            reminder_list = []
            for reminder in reminders:
                due_date = datetime.datetime.fromisoformat(reminder['due_date'].replace('Z', '+00:00'))
                due_str = due_date.strftime('%A, %B %d at %I:%M %p')
                
                # Check if overdue
                if due_date < now and not reminder['completed']:
                    status = "OVERDUE"
                elif reminder['completed']:
                    status = "✓"
                else:
                    status = ""
                
                reminder_list.append(f"- {due_str}: {reminder['content']} {status}")
            
            return "Your Reminders:\n" + "\n".join(reminder_list)
            
        except Exception as e:
            return f"Error retrieving reminders: {str(e)}"
```

### 2. Update Configuration File

Add notes settings in your `config.yaml`:

```yaml
# Service integrations
services:
  # Other services...
  
  notes:
    enabled: true
    notes_file: "notes.json"
    reminders_file: "reminders.json"
```

### 3. Add Command Handlers

Update `main.py` to handle notes and reminders commands:

```python
# Inside the main loop where commands are processed
elif user_input.lower().startswith('note '):
    if config['services'].get('notes', {}).get('enabled', False):
        note_content = user_input[5:].strip()
        if note_content:
            from friday.services.notes import NotesService
            
            notes_service = NotesService(
                config['services']['notes']['notes_file'],
                config['services']['notes']['reminders_file']
            )
            
            result = await notes_service.add_note(note_content)
            print(result)
            tts.generate_speech(result)
        else:
            print("Please specify note content after 'note'")

elif user_input.lower() in ['notes', 'show notes', 'get notes']:
    if config['services'].get('notes', {}).get('enabled', False):
        from friday.services.notes import NotesService
        
        notes_service = NotesService(
            config['services']['notes']['notes_file'],
            config['services']['notes']['reminders_file']
        )
        
        notes = await notes_service.get_notes()
        print(notes)
        tts.generate_speech(notes)

elif user_input.lower().startswith('remind me to '):
    if config['services'].get('notes', {}).get('enabled', False):
        # Try to parse "remind me to X on/at Y" format
        import re
        
        # Match patterns like "remind me to call mom on 2023-01-15" or "remind me to take medicine at 15:00"
        match = re.search(r'remind me to (.+) (?:on|at) (.+)', user_input, re.IGNORECASE)
        
        if match:
            task = match.group(1).strip()
            time_str = match.group(2).strip()
            
            from friday.services.notes import NotesService
            
            notes_service = NotesService(
                config['services']['notes']['notes_file'],
                config['services']['notes']['reminders_file']
            )
            
            result = await notes_service.add_reminder(task, time_str)
            print(result)
            tts.generate_speech(result)
        else:
            print("Please use format: remind me to [task] on/at [date/time]")

elif user_input.lower() in ['reminders', 'show reminders', 'get reminders']:
    if config['services'].get('notes', {}).get('enabled', False):
        from friday.services.notes import NotesService
        
        notes_service = NotesService(
            config['services']['notes']['notes_file'],
            config['services']['notes']['reminders_file']
        )
        
        reminders = await notes_service.get_reminders()
        print(reminders)
        tts.generate_speech(reminders)
```

## Best Practices for Integrations

1. **Make integrations optional**: Always check if the service is enabled before trying to use it
2. **Handle errors gracefully**: Provide helpful error messages if an integration fails
3. **Secure API keys**: Store sensitive API keys securely
4. **Use async functions**: Keep Friday responsive while waiting for external APIs
5. **Provide clear documentation**: Make it easy for users to set up integrations
6. **Respect rate limits**: Be mindful of API rate limits for external services

These integration examples demonstrate how to extend Friday's capabilities beyond just conversation. By adding these services, Friday becomes a more powerful personal assistant that can interact with the real world. 