for (let i = 0; i < 50; i++) {
 fetch("http://localhost:5000/api/user/profile", {
  headers:{ Authorization:"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI2OWFkMmU3YjYzMzYzODg1NTY2Yjk0MWQiLCJyb2xlIjoidXNlciIsImlhdCI6MTc3Mjk2ODI2NywiZXhwIjoxNzcyOTY5MTY3fQ.X4XNcQ3AaNY69dGAZlSCfCQzKxVubuwml9tuPdN_kF0"}
 });
}