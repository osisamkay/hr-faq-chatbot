// src/components/Dashboard.js
import React, { useEffect, useState } from "react";
import axios from "axios";
import { Box, Typography, Paper, CircularProgress } from "@mui/material";

const Dashboard = () => {
   const [profile, setProfile] = useState(null);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await axios.get("http://localhost:5000/user_profile/john_doe");
        setProfile(res.data);
      } catch (error) {
        console.error("Error:", error);
      }
    };
    fetchProfile();
  }, []);

  useEffect(() => {
    const fetchProfile = async () => {
      // Simulate fetching data for testing
      setTimeout(() => {
        setProfile({
          employment_status: "Full-time",
          vacation_days: 20,
        });
      }, 1000);
    };
    fetchProfile();
  }, []);

  return (
    <Paper
      elevation={3}
      style={{
        padding: "20px",
        margin: "20px auto",
        maxWidth: "600px",
        textAlign: "center",
      }}
    >
      <Typography variant="h4" gutterBottom>
        Personalized Dashboard
      </Typography>
      {profile ? (
        <Box display="flex" flexDirection="column" gap={2}>
          <Typography variant="body1">
            <strong>Employment Status:</strong> {profile.employment_status}
          </Typography>
          <Typography variant="body1">
            <strong>Vacation Days:</strong> {profile.vacation_days}
          </Typography>
        </Box>
      ) : (
        <CircularProgress />
      )}
    </Paper>
  );
};

export default Dashboard;
