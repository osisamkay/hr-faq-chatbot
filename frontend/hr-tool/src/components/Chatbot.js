import React, { useState } from "react";
import axios from "axios";
import { Box, Button, TextField, Typography, Paper } from "@mui/material";

const Chatbot = () => {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    setLoading(true);
    if (!message.trim()) {
      alert("Please enter a question!");
      return;
    }

    try {
      const res = await axios.post("https://hr-faq-chatbot.onrender.com/chatbot", {
        message: message,
      });

      setResponse(res.data.response);
      setLoading(false)
    } catch (error) {
      console.error("Error sending message:", error);
      setLoading(false)
      setResponse("An error occurred while searching the document.");
    }
  };

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
        Interactive FAQ Chatbot with PDF Search
      </Typography>
      <Box display="flex" flexDirection="column" gap={2}>
        <TextField
          label="Ask a question about the HR policies"
          variant="outlined"
          multiline
          rows={4}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />
        <Button
          variant="contained"
          color="primary"
          onClick={sendMessage}
          size="large"
        >
          {loading ? "Searching..." : "Send"}
        </Button>
        {response && (
          <Typography
            variant="body1"
            style={{
              marginTop: "20px",
              padding: "10px",
              border: "1px solid #ddd",
              borderRadius: "4px",
              backgroundColor: "#f9f9f9",
            }}
          >
            {response}
          </Typography>
        )}
      </Box>
    </Paper>
  );
};

export default Chatbot;
