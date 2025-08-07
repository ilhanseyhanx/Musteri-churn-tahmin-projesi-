const express = require("express");
const cors = require("cors");
const path = require("path");

const app = express();
const PORT = 3000;

app.use(cors());

app.use(express.static("."));

app.get("/",(req, res) =>  {
    res.sendFile(path.join(__dirname,"index.html"));
});

app.listen(PORT,() => {
    console.log(`Frontend server çalışıyor: http://localhost:${PORT}`);
})