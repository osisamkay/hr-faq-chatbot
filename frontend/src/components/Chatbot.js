import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  CircularProgress,
} from "@mui/material";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const lastMessageRef = useRef(null);

  // const Base_Url="http://127.0.0.1:10000"
  const Base_Url="https://osisamkay-hr-faq-tool.hf.space"

  const sendMessage = async () => {
    if (!inputMessage.trim()) {
      alert("Please enter a message!");
      return;
    }

    const newMessages = [
      ...messages,
      { sender: "user", text: inputMessage },
    ];
    setMessages(newMessages);
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await axios.post(
        `${Base_Url}/chatbot`, 
        {
          message: inputMessage,
          username: "guest", 
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const botResponse = response.data.response;
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: botResponse },
      ]);
      console.log(botResponse);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: "Sorry, there was an error. Please try again!" },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  };

 useEffect(() => {
    if (lastMessageRef.current) {
      lastMessageRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [messages]);

  return (
    <Paper
      elevation={3}
      sx={{
        padding: "20px",
        maxWidth: "600px",
        margin: "20px auto",
        borderRadius: "15px",
        boxShadow: 3,
      }}
    >
      <Typography
        variant="h5"
        gutterBottom
        align="center"
        sx={{ fontWeight: "bold" }}
      >
         Interactive FAQ Chatbot with PDF Search
      </Typography>

      {/* Chat Window */}
      <Box
        
        sx={{
          border: "1px solid #ccc",
          borderRadius: "10px",
          padding: "10px",
          maxHeight: "400px",
          overflowY: "scroll",
          marginBottom: "20px",
          backgroundColor: "#f9f9f9",
          position: "relative",
        }}
      >
        <Typography
          variant="subtitle2"
          align="center"
          sx={{
            backgroundColor: "#fff",
            padding: "10px",
            color: "#888",
          }}
        >
          Chat Conversation
        </Typography>
        {messages.length === 0 ? (
          <Typography
            variant="body2"
            align="center"
            sx={{
              color: "#888",
              marginTop: "20px",
            }}
          >
            Start a conversation by asking a question!
          </Typography>
        ) : (
          messages.map((msg, index) => (
            <Box
            ref={index === messages.length - 1 ? lastMessageRef : null}
              key={index}
              sx={{
                display: "flex",
                justifyContent:
                  msg.sender === "user" ? "flex-end" : "flex-start",
                marginBottom: "10px",
              }}
            >
              <Box
                sx={{
                  padding: "10px",
                  borderRadius: "10px",
                  maxWidth: "70%",
                  backgroundColor:
                    msg.sender === "user" ? "#1976d2" : "#e0e0e0",
                  color: msg.sender === "user" ? "#fff" : "#000",
                  wordBreak: "break-word",
                }}
              >
                {msg.text}
              </Box>
            </Box>
          ))
        )}
      </Box>

      {/* Input Field and Button */}
      <Box
        sx={{
          display: "flex",
          gap: "10px",
        }}
      >
        <TextField
          fullWidth
          variant="outlined"
          label="Type your question..."
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <Button
          variant="contained"
          color="primary"
          onClick={sendMessage}
          disabled={isLoading}
          sx={{
            minWidth: "100px",
          }}
        >
          {isLoading ? <CircularProgress size={24} color="inherit" /> : "Send"}
        </Button>
      </Box>
    </Paper>
  );
};

export default Chatbot;
