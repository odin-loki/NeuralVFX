// The viewer's scene mode (docs/VIEWER.md): a scene script edited live next to the scene it plays. The script window
// (open, save, an editor with line numbers, the error at its line), the picture with play, pause, scrub, restart and
// speed, the script's inputs as sliders, a button per named rule, a field probe under the pointer, and the effects'
// folder. The work is SceneSession's; this draws it with Dear ImGui.
#pragma once

#include "scene_session.hpp"
#include "texture.hpp"

#include <imgui.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nfx::viewer {

class SceneUi {
 public:
  struct Options {
    std::string script_path;  // opened at start (empty: none)
    SessionSettings settings;
    double start_at = 0;      // seconds the first build plays to
    bool paused = false;
  };
  explicit SceneUi(Options o);

  // One UI frame (between ImGui::NewFrame and ImGui::Render): plays the scene by dt real seconds (times the speed),
  // draws its picture and the mode's windows in the rectangle (x, y, w, h) of the screen.
  void frame(float x, float y, float w, float h, double dt);

  SceneSession& session() { return session_; }
  bool open_file(const std::string& path, double at = 0.0);
  bool save_file(const std::string& path);
  // An edit as if typed: the first `from` in the text becomes `to` (tests). False if `from` is not there.
  bool replace_text(const std::string& from, const std::string& to);
  // The bus overlay: 0 none, 1 heat, 2 soot.
  void set_overlay(int field) { overlay_field_ = field; }
  // Pins the field probe at a pixel of the picture (as a click does).
  void pin(float px, float py) {
    pinned_ = true;
    pin_px_ = {px, py};
  }
  // Where the picture was drawn in the last frame, in screen pixels (false: not drawn).
  bool picture_rect(ImVec2& min, ImVec2& max) const;

 private:
  void script_window(float x, float y, float w, float h);
  void editor(float height);
  void scene_window(float x, float y, float w, float h);
  void controls_window(float x, float y, float w, float h);
  void watch_file();
  void update_overlay();
  static int callback(ImGuiInputTextCallbackData* data);

  SceneSession session_;
  std::string text_;                     // the editor's buffer
  std::string path_;                     // the file it was opened from or saved to
  std::string path_field_;               // the path in the script window
  std::string dir_field_;                // the effects folder in the controls window
  bool modified_ = false;                // unsaved edits
  std::filesystem::file_time_type disk_time_{};  // the file's time when read or written
  bool disk_changed_ = false;            // changed on disk while there were unsaved edits
  Clock::time_point last_watch_{};
  std::string note_;                     // the last file message
  bool playing_ = true;
  float speed_ = 1.f;
  bool scrubbing_ = false;
  bool loop_ = true;                     // back to 0 at the end of the script's length
  // the editor
  std::vector<int> line_starts_;         // byte offset of each line
  int cursor_ = 0, cursor_line_ = 1, last_cursor_line_ = 1;
  int goto_offset_ = -1;                 // move the cursor here (a jump to the error, an edit made by a test)
  int scroll_line_ = 0;                  // scroll to show this line (0: none)
  int scroll_wait_ = 0;                  // frames to wait before scrolling
  // the picture
  Texture picture_{true}, overlay_{true};
  ImVec2 pic_min_{}, pic_max_{};
  bool pic_drawn_ = false;
  int overlay_field_ = 0;                // 0 none, 1 heat, 2 soot
  std::vector<float> overlay_values_;
  std::vector<std::uint8_t> overlay_rgba_;
  int overlay_w_ = 0, overlay_h_ = 0;
  // the probe
  bool hover_ = false, pinned_ = false;
  ImVec2 hover_px_{}, pin_px_{};
  struct Probe {
    bool ok = false;
    float x = 0, y = 0;
    nvfx_scene_fields f{};
  } hover_probe_, pin_probe_;
  std::string trigger_note_;
  // cost
  double frame_ms_ = 0;
};

}  // namespace nfx::viewer
