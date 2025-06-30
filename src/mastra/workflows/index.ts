import { anthropic } from '@ai-sdk/anthropic';
import { Agent } from '@mastra/core/agent';
import { createStep, createWorkflow } from '@mastra/core/workflows';
import { z } from 'zod';

const llm = anthropic(process.env.MODEL ?? "claude-3-5-sonnet-20240620");

const agent = new Agent({
  name: 'Weather Agent',
  model: llm,
  instructions: `
        You are a local activities and travel expert who excels at weather-based planning. Analyze the weather data and provide practical activity recommendations.

        For each day in the forecast, structure your response exactly as follows:

        📅 [Day, Month Date, Year]
        ═══════════════════════════

        🌡️ WEATHER SUMMARY
        • Conditions: [brief description]
        • Temperature: [X°C/Y°F to A°C/B°F]
        • Precipitation: [X% chance]

        🌅 MORNING ACTIVITIES
        Outdoor:
        • [Activity Name] - [Brief description including specific location/route]
          Best timing: [specific time range]
          Note: [relevant weather consideration]

        🌞 AFTERNOON ACTIVITIES
        Outdoor:
        • [Activity Name] - [Brief description including specific location/route]
          Best timing: [specific time range]
          Note: [relevant weather consideration]

        🏠 INDOOR ALTERNATIVES
        • [Activity Name] - [Brief description including specific venue]
          Ideal for: [weather condition that would trigger this alternative]

        ⚠️ SPECIAL CONSIDERATIONS
        • [Any relevant weather warnings, UV index, wind conditions, etc.]

        Guidelines:
        - Suggest 2-3 time-specific outdoor activities per day
        - Include 1-2 indoor backup options
        - For precipitation >50%, lead with indoor activities
        - All activities must be specific to the location
        - Include specific venues, trails, or locations
        - Consider activity intensity based on temperature
        - Keep descriptions concise but informative

        Maintain this exact formatting for consistency, using the emoji and section headers as shown.
      `,
});

const forecastSchema = z.object({
  date: z.string(),
  maxTemp: z.number(),
  minTemp: z.number(),
  precipitationChance: z.number(),
  condition: z.string(),
  location: z.string(),
});

function getWeatherCondition(code: number): string {
  const conditions: Record<number, string> = {
    0: 'Clear sky',
    1: 'Mainly clear',
    2: 'Partly cloudy',
    3: 'Overcast',
    45: 'Foggy',
    48: 'Depositing rime fog',
    51: 'Light drizzle',
    53: 'Moderate drizzle',
    55: 'Dense drizzle',
    61: 'Slight rain',
    63: 'Moderate rain',
    65: 'Heavy rain',
    71: 'Slight snow fall',
    73: 'Moderate snow fall',
    75: 'Heavy snow fall',
    95: 'Thunderstorm',
  };
  return conditions[code] || 'Unknown';
}

const fetchWeather = createStep({
  id: 'fetch-weather',
  description: 'Fetches weather forecast for a given city',
  inputSchema: z.object({
    city: z.string().describe('The city to get the weather for'),
  }),
  outputSchema: forecastSchema,
  execute: async ({ inputData }) => {
    if (!inputData) {
      throw new Error('Input data not found');
    }

    const geocodingUrl = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(inputData.city)}&count=1`;
    const geocodingResponse = await fetch(geocodingUrl);
    const geocodingData = (await geocodingResponse.json()) as {
      results: { latitude: number; longitude: number; name: string }[];
    };

    if (!geocodingData.results?.[0]) {
      throw new Error(`Location '${inputData.city}' not found`);
    }

    const { latitude, longitude, name } = geocodingData.results[0];

    const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=precipitation,weathercode&timezone=auto,&hourly=precipitation_probability,temperature_2m`;
    const response = await fetch(weatherUrl);
    const data = (await response.json()) as {
      current: {
        time: string;
        precipitation: number;
        weathercode: number;
      };
      hourly: {
        precipitation_probability: number[];
        temperature_2m: number[];
      };
    };

    const forecast = {
      date: new Date().toISOString(),
      maxTemp: Math.max(...data.hourly.temperature_2m),
      minTemp: Math.min(...data.hourly.temperature_2m),
      condition: getWeatherCondition(data.current.weathercode),
      precipitationChance: data.hourly.precipitation_probability.reduce((acc, curr) => Math.max(acc, curr), 0),
      location: inputData.city,
    };

    return forecast;
  },
});

const planActivities = createStep({
  id: 'plan-activities',
  description: 'Suggests activities based on weather conditions',
  inputSchema: forecastSchema,
  outputSchema: z.object({
    activities: z.string(),
  }),
  execute: async ({ inputData }) => {
    const forecast = inputData;

    if (!forecast) {
      throw new Error('Forecast data not found');
    }

    const prompt = `Based on the following weather forecast for ${forecast.location}, suggest appropriate activities:
      ${JSON.stringify(forecast, null, 2)}
      `;

    const response = await agent.stream([
      {
        role: 'user',
        content: prompt,
      },
    ]);

    let activitiesText = '';

    for await (const chunk of response.textStream) {
      process.stdout.write(chunk);
      activitiesText += chunk;
    }

    return {
      activities: activitiesText,
    };
  },
});

const weatherWorkflow = createWorkflow({
  id: 'weather-workflow',
  inputSchema: z.object({
    city: z.string().describe('The city to get the weather for'),
  }),
  outputSchema: z.object({
    activities: z.string(),
  }),
})
  .then(fetchWeather)
  .then(planActivities);

weatherWorkflow.commit();

// New Web Search Workflow using Anthropic's built-in web search
const webSearchAgent = new Agent({
  name: 'Web Search Activity Planner',
  model: anthropic(process.env.MODEL ?? "claude-4-sonnet-20250514"),
  defaultStreamOptions: {
    providerOptions: {
      anthropic: {
        sendReasoning: true,
        thinking: { type: 'enabled', budgetTokens: 8000 },
        webSearch: {
          maxUses: 3,
        },
      }
    },
  },
  defaultGenerateOptions: {
    providerOptions: {
      anthropic: {
        sendReasoning: true,
        thinking: { type: 'enabled', budgetTokens: 8000 },
        webSearch: {
          maxUses: 3,
        },
      }
    },
  },
  instructions: `
    You are an expert activity planner with access to current web information.
    
    Your task is to:
    1. Use web search to find current, relevant information about the location/topic
    2. Search for current events, weather, attractions, activities, and local insights
    3. Plan comprehensive activities based on your web search findings
    
    When planning activities:
    - Use web search to find current information about attractions, events, weather
    - Search for local recommendations, reviews, and current operating hours
    - Look for seasonal activities, special events, or festivals
    - Consider current local conditions and accessibility
    - Provide specific, actionable recommendations with details
    
    Structure your response as:
    
    🔍 RESEARCH SUMMARY
    • Key findings from web search
    • Current conditions and notable information
    
    📍 LOCATION OVERVIEW
    • Brief description of the area
    • Current weather/seasonal considerations
    
    🎯 RECOMMENDED ACTIVITIES
    
    🌅 MORNING ACTIVITIES
    • [Activity Name] - [Description with specific details from web search]
      📍 Location: [Specific address/area]
      ⏰ Best time: [Time range]
      💡 Tip: [Current info from web search]
    
    🌞 AFTERNOON ACTIVITIES  
    • [Activity Name] - [Description with specific details]
      📍 Location: [Specific address/area]
      ⏰ Best time: [Time range]
      💡 Tip: [Current info from web search]
    
    🌆 EVENING ACTIVITIES
    • [Activity Name] - [Description with specific details]
      📍 Location: [Specific address/area]
      ⏰ Best time: [Time range]
      💡 Tip: [Current info from web search]
    
    🏠 BACKUP/INDOOR OPTIONS
    • [Activity Name] - [Description]
      📍 Location: [Specific venue]
      💡 Good for: [Weather conditions or circumstances]
    
    ⚠️ CURRENT CONSIDERATIONS
    • Any current events, closures, or special conditions found via web search
    • Booking requirements or advance planning needed
    • Transportation or accessibility notes
  `,
});

const webSearchInfo = createStep({
  id: 'web-search-info',
  description: 'Uses web search to gather current information about a location/topic',
  inputSchema: z.object({
    query: z.string().describe('The location or topic to search for'),
    focus: z.string().optional().describe('Specific focus area (e.g., activities, events, attractions)'),
  }),
  outputSchema: z.object({
    searchResults: z.string().describe('Summary of web search findings'),
    query: z.string(),
    focus: z.string().optional(),
  }),
  execute: async ({ inputData }) => {
    if (!inputData) {
      throw new Error('Input data not found');
    }

    const searchQuery = inputData.focus 
      ? `${inputData.query} ${inputData.focus} current information activities attractions events`
      : `${inputData.query} current activities attractions events what to do`;

    console.log(`🔍 Searching for: ${searchQuery}`);

    const response = await webSearchAgent.generate([
      {
        role: 'user',
        content: `Search for current information about: ${searchQuery}
        
Please use web search to find the most up-to-date information and provide a comprehensive summary of what you discover.`,
      },
    ]);

    return {
      searchResults: response.text,
      query: inputData.query,
      focus: inputData.focus,
    };
  },
});

const planWebSearchActivities = createStep({
  id: 'plan-web-search-activities',
  description: 'Plans activities based on web search findings',
  inputSchema: z.object({
    searchResults: z.string(),
    query: z.string(),
    focus: z.string().optional(),
  }),
  outputSchema: z.object({
    activities: z.string(),
  }),
  execute: async ({ inputData }) => {
    if (!inputData) {
      throw new Error('Search results not found');
    }

    console.log(`🎯 Planning activities for: ${inputData.query}`);

    const prompt = `Based on the web search findings below, create a detailed activity plan:

${inputData.searchResults}

Please create a comprehensive activity plan using the structured format in your instructions. Make sure to incorporate the specific, current information you found through web search.`;

    const response = await webSearchAgent.stream([
      {
        role: 'user',
        content: prompt,
      },
    ]);

    let activitiesText = '';
    for await (const chunk of response.textStream) {
      process.stdout.write(chunk);
      activitiesText += chunk;
    }

    return {
      activities: activitiesText,
    };
  },
});

const webSearchWorkflow = createWorkflow({
  id: 'web-search-workflow',
  inputSchema: z.object({
    query: z.string().describe('The location or topic to search for and plan activities'),
    focus: z.string().optional().describe('Specific focus area (e.g., outdoor activities, cultural events, food scene)'),
  }),
  outputSchema: z.object({
    activities: z.string(),
  }),
})
  .then(webSearchInfo)
  .then(planWebSearchActivities);

webSearchWorkflow.commit();

export { weatherWorkflow, webSearchWorkflow };
