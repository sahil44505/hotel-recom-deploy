import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, sin, cos, sqrt, atan2
import requests
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["https://hotel-deploy-five.vercel.app"]}})

@app.route('/')
def health_check():
    return "API is running!", 200

@app.route('/api/recommendations', methods=['GET'])
def recommendations():
    print("sending")
    try:
        # Fetch booking data
        bookings_response = requests.get("https://hotel-deploy-five.vercel.app/api/getbookingsforpy")
        if bookings_response.status_code != 200:
            return jsonify([])

        bookings_data = bookings_response.json()
        df_bookings = pd.DataFrame(bookings_data)

        worldwide_results = []

        def parse_price(price_str):
            try:
                numeric = re.sub(r'[^\d.]', '', price_str)
                return float(numeric) if numeric else None
            except Exception:
                return None

        # Fetch hotel data for each booking title
        for title in df_bookings['title'].unique():
            search_payload = {"title": title}
            response = requests.post("https://hotel-deploy-five.vercel.app/api/searchhotelsforpy", json=search_payload)

            if response.status_code == 200:
                hotels_data = response.json()
                if isinstance(hotels_data, dict):
                    results = hotels_data.get("ads") or hotels_data.get("data", [])
                elif isinstance(hotels_data, list):
                    results = hotels_data
                else:
                    results = []

                for hotel in results[1:]:  # skip the first (assumed to be the current hotel)
                    if not isinstance(hotel, dict):
                        hotel = {"url": hotel}

                    hotel['booking_title'] = title
                    hotel['title'] = (hotel.get("title") or hotel.get("name", "")).strip()

                    # Price
                    if "extracted_price" in hotel:
                        hotel['price'] = hotel["extracted_price"]
                    elif "price" in hotel:
                        parsed = parse_price(hotel["price"])
                        hotel['price'] = parsed if parsed is not None else 0
                    elif isinstance(hotel.get("rate_per_night"), dict):
                        hotel['price'] = hotel["rate_per_night"].get("extracted_lowest", 0)
                    elif isinstance(hotel.get("total_rate"), dict):
                        hotel['price'] = hotel["total_rate"].get("extracted_lowest", 0)
                    else:
                        hotel['price'] = 0

                    # Rating
                    hotel['rating'] = float(hotel.get("rating") or hotel.get("extracted_hotel_class") or 0)

                    # GPS
                    gps = hotel.get("gps_coordinates", {"latitude": 0, "longitude": 0})
                    hotel['gps_coordinates'] = gps if isinstance(gps, dict) else {"latitude": 0, "longitude": 0}

                    worldwide_results.append(hotel)

        df_worldwide = pd.DataFrame(worldwide_results)

        if df_worldwide.empty:
            return jsonify([])

        # Compute averages from user's bookings
        user_avg_rating = df_bookings['ratings'].mean()
        user_avg_price = df_bookings['totalPrice'].mean()
        user_avg_lat = df_bookings['gps_coordinates'].apply(lambda x: x.get('latitude', 0)).mean()
        user_avg_lon = df_bookings['gps_coordinates'].apply(lambda x: x.get('longitude', 0)).mean()
        user_avg_gps = {"latitude": user_avg_lat, "longitude": user_avg_lon}
        user_title = df_bookings['title'].mode()[0]

        # Text similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_world = vectorizer.fit_transform(df_worldwide['title'].astype(str).tolist())
        tfidf_user = vectorizer.transform([user_title])
        df_worldwide['title_similarity'] = cosine_similarity(tfidf_user, tfidf_world)[0]

        # Rating similarity
        df_worldwide['rating_similarity'] = 1 - (abs(df_worldwide['rating'] - user_avg_rating) / 5)
        df_worldwide['rating_similarity'] = df_worldwide['rating_similarity'].clip(0, 1)

        # Price similarity
        df_worldwide['price_similarity'] = 1 - (abs(df_worldwide['price'] - user_avg_price) / user_avg_price)
        df_worldwide['price_similarity'] = df_worldwide['price_similarity'].clip(0, 1)

        # GPS similarity
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        def gps_similarity(hotel_gps, user_gps):
            distance = haversine(user_gps['latitude'], user_gps['longitude'],
                                 hotel_gps.get('latitude', 0), hotel_gps.get('longitude', 0))
            return 1 / (1 + distance)

        df_worldwide['gps_similarity'] = df_worldwide['gps_coordinates'].apply(
            lambda gps: gps_similarity(gps, user_avg_gps))

        # Combine similarities
        w_title = 0.2
        w_rating = 0.2
        w_price = 0.2
        w_gps = 0.2

        df_worldwide['overall_score'] = (w_title * df_worldwide['title_similarity'] +
                                         w_rating * df_worldwide['rating_similarity'] +
                                         w_price * df_worldwide['price_similarity'] +
                                         w_gps * df_worldwide['gps_similarity'])

        top5 = df_worldwide.sort_values('overall_score', ascending=False)
        top5_records = top5[['title', 'rating', 'price', 'gps_coordinates', 'overall_score']].to_dict(orient='records')

        return jsonify(top5_records[:5])

    except Exception as e:
        print("Error in recommendations:", str(e))
        return jsonify([]), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    
    # host='0.0.0.0' is the critical fix for Render's port scanning
    app.run(host='0.0.0.0', port=port)
