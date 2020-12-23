"""
Create a billboard api that can fetch data form the billboard api

- authorisation must be added containing rapid-api-key

"""
import requests
import json

with open("authorisation.json", "r") as auth_file:
    auth_data = auth_file.read()

credentials = json.loads(auth_data)
rapid_api_key_from_cred = credentials["x-rapidapi-key"]

class BillboardScraper():
    rapid_api_key = ""

    def __init__(self, rapid_api_key=rapid_api_key_from_cred):
        """
        Init the service with the rapid api key
        """
        super().__init__()
        self.rapid_api_key = rapid_api_key
        
    def get_api_headers(self):
        """
        Returns the api headers to be used
        """
        return {
            "x-rapidapi-key": self.rapid_api_key,
            "x-rapidapi-host": "billboard-api2.p.rapidapi.com",
        }

    def get_chart_for_date(self, date):
        """
        Get the chart for the given date
        """
        if (date == None):
            print("Date must be provided")
            return None

        #TODO: Add checks that the date is in the correct format to be useds

        chart_endpoint = "https://billboard-api2.p.rapidapi.com/hot-100?date=" + str(date)

        try:
            r = requests.get(chart_endpoint, headers=self.get_api_headers())
            if (r.status_code in range(200,299)):
                print("Request finished successfully")
                r_data = r.json()
                print(r_data)
            else:
                print("error")
                print(r.json())

        except Exception as e:
            print(e)


scrape = BillboardScraper()
scrape.get_chart_for_date("2019-05-11")