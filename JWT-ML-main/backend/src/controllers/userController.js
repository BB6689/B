import User from "../models/User.js";

export const profile = async (req, res) => {
  try {

    const userId = req.user.id;

    const user = await User.findById(userId).select("-passwordHash");

    if (!user) {
      return res.status(404).json({
        message: "User not found"
      });
    }

    return res.status(200).json({
      message: "Profile fetched ✅",
      user
    });

  } catch (err) {

    return res.status(500).json({
      message: "Server error",
      error: err.message
    });

  }
};