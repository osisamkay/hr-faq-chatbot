// src/App.js
import React from "react";
import Chatbot from "./components/Chatbot";
import Dashboard from "./components/Dashboard";
import { Box, Typography } from "@mui/material";

function App() {
  return (
    <Box
      style={{
        backgroundColor: "#f5f5f5",
        minHeight: "100vh",
        padding: "20px",
        textAlign: "center",
      }}
    >
      <Typography variant="h3" gutterBottom color="primary">
        HR Onboarding Tool
      </Typography>
      <Chatbot />
      <Dashboard />
    </Box>
  );
}

export default App;
