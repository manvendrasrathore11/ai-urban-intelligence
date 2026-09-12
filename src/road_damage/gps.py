# ============================================================
# GPS PROVIDER
# ============================================================

class GPSProvider:
    """
    Provides GPS coordinates for each video frame.

    For now we use simulated GPS data so that we can
    develop and test the complete architecture.

    Later this class can be replaced by a real GPS source.
    """

    def __init__(
        self,
        start_latitude=26.9124,
        start_longitude=75.7873,
        latitude_step=0.000001,
        longitude_step=0.000001
    ):

        self.start_latitude = start_latitude
        self.start_longitude = start_longitude

        self.latitude_step = latitude_step
        self.longitude_step = longitude_step

    def get_location(self, frame_number):

        latitude = (
            self.start_latitude
            + frame_number * self.latitude_step
        )

        longitude = (
            self.start_longitude
            + frame_number * self.longitude_step
        )

        return {
            "latitude": round(latitude, 7),
            "longitude": round(longitude, 7)
        }