const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI2OWFkNjQ3ZWQ5YjIwZjczZGU5NjIzNGQiLCJyb2xlIjoidXNlciIsImlhdCI6MTc3Mjk3MjIxMSwiZXhwIjoxNzcyOTczMTExfQ.UNHCeBOEODN89jgGKz5lu3wLB5Yc_UmclCs_FV6H-pQ";
const URL = "http://localhost:5000/api/user/profile";

async function flood() {
  console.log("Starting flood attack simulation...");

  for (let i = 0; i < 30; i++) {
    fetch(URL, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then(() => console.log("Request sent:", i))
      .catch((err) => console.log("Error:", err.message));
  }
}

flood();