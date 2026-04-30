const endpoints = [
 "/api/user/profile",
 "/api/events",
 "/api/admin/logs",
 "/api/admin/alerts",
 "/api/test",
 "/api/settings",
 "/api/random"
];

endpoints.forEach(e => {
 fetch("http://localhost:5000"+e, {
  headers:{ Authorization:"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI2OWFkNWU3ODYyYWMxMTQyNDMwZDBkM2UiLCJyb2xlIjoidXNlciIsImlhdCI6MTc3Mjk3MDEwNywiZXhwIjoxNzcyOTcxMDA3fQ.CK3K5jzMQk-j0OKqlZtGLaBD26E__ySnMAGaEMBE-Dw"}
 });
});