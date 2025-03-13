import googlemaps
import numpy as np
import os

# Google Maps API key (should be stored securely in production)
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "API")

# Initialize the Google Maps client
gmaps = googlemaps.Client(key=API_KEY)

def get_travel_time(origin, destination):
    """
    Get travel time between two locations using Google Maps API.
    
    Args:
        origin (tuple): Coordinates of origin (lat, lon)
        destination (tuple): Coordinates of destination (lat, lon)
        
    Returns:
        float: Travel time in minutes
    """
    try:
        # Request directions
        directions_result = gmaps.directions(
            origin,
            destination,
            mode="driving",  # Can be changed to other modes like "walking", "bicycling", etc.
            departure_time="now"  # Use current time for traffic conditions
        )
        
        # Extract duration in seconds and convert to minutes
        if directions_result and len(directions_result) > 0:
            duration_seconds = directions_result[0]['legs'][0]['duration']['value']
            return duration_seconds / 60  # Convert to minutes
        else:
            # Fallback to Euclidean distance if API call fails
            print("Warning: Could not get travel time from Google Maps API, using Euclidean distance instead.")
            return fallback_distance(origin, destination)
            
    except Exception as e:
        print(f"Error getting travel time from Google Maps API: {e}")
        return fallback_distance(origin, destination)

def fallback_distance(origin, destination):
    """
    Calculate Euclidean distance as a fallback when API fails.
    
    Args:
        origin (tuple): Coordinates of origin (lat, lon)
        destination (tuple): Coordinates of destination (lat, lon)
        
    Returns:
        float: Estimated travel time in minutes based on Euclidean distance
    """
    # Calculate Euclidean distance
    origin_array = np.array(origin)
    destination_array = np.array(destination)
    distance = np.linalg.norm(origin_array - destination_array)
    
    # Convert to minutes (assuming 60 units per hour)
    travel_time = (distance / 60) * 60
    
    return travel_time

def create_distance_matrix_with_api(locations):
    """
    Create a distance matrix using Google Maps API.
    
    Args:
        locations (list): List of location coordinates (lat, lon)
        
    Returns:
        np.ndarray: Matrix of travel times between locations in minutes
    """
    n_locations = len(locations)
    distance_matrix = np.zeros((n_locations, n_locations))
    
    for i in range(n_locations):
        for j in range(n_locations):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                distance_matrix[i, j] = get_travel_time(locations[i], locations[j])
    
    return distance_matrix