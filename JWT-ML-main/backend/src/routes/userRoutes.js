import express from "express";
import { profile } from "../controllers/userController.js";

const router = express.Router();
router.get("/profile", profile);

router.get("/test", (req, res) => {
  res.json({ message: "user route working" });
});

export default router;