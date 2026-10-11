// The viewer's scene mode: see scene_ui.hpp.
#include "scene_ui.hpp"

#include <imgui_internal.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <sstream>

namespace nfx::viewer {

namespace {

constexpr ImU32 kErrorLine = IM_COL32(210, 55, 45, 80);
constexpr ImU32 kErrorMark = IM_COL32(255, 110, 90, 255);
constexpr ImVec4 kErrorText{1.f, 0.48f, 0.42f, 1.f};
constexpr ImVec4 kBusyText{1.f, 0.85f, 0.4f, 1.f};
constexpr ImVec4 kGoodText{0.55f, 0.9f, 0.55f, 1.f};
constexpr ImVec4 kEditorBg{0.075f, 0.085f, 0.11f, 1.f};
constexpr ImGuiWindowFlags kFixed = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings;

std::optional<std::string> read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::nullopt;
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// A slider's range for an input from its starting value (ctrl+click on the slider types any value).
void input_range(float v, float& lo, float& hi) {
  if (v >= 0.f && v <= 1.f) {
    lo = 0.f, hi = 1.f;
  } else if (v > 1.f) {
    lo = 0.f, hi = 2.f * v;
  } else if (v >= -1.f) {
    lo = -1.f, hi = 1.f;
  } else {
    lo = 2.f * v, hi = -2.f * v;
  }
}

void smooth(double& avg, double v) { avg = avg == 0.0 ? v : 0.9 * avg + 0.1 * v; }

}  // namespace

SceneUi::SceneUi(Options o) : session_(o.settings), dir_field_(o.settings.effects_dir), playing_(!o.paused) {
  if (!o.script_path.empty()) open_file(o.script_path, o.start_at);
}

// --- files --------------------------------------------------------------------------------------------------------

bool SceneUi::open_file(const std::string& path, double at) {
  path_field_ = path;
  const auto text = read_text(path);
  if (!text) {
    note_ = std::format("cannot read {}", path);
    return false;
  }
  text_ = *text;
  path_ = path;
  modified_ = false;
  disk_changed_ = false;
  std::error_code ec;
  disk_time_ = std::filesystem::last_write_time(path, ec);
  session_.open(text_, std::filesystem::path(path).filename().string(), at);
  note_ = std::format("opened {}", path);
  return true;
}

bool SceneUi::save_file(const std::string& path) {
  if (path.empty()) {
    note_ = "give the file a path first";
    return false;
  }
  {
    std::ofstream f(path, std::ios::binary);
    f << text_;
    if (!f) {
      note_ = std::format("cannot write {}", path);
      return false;
    }
  }
  path_ = path;
  modified_ = false;
  disk_changed_ = false;
  std::error_code ec;
  disk_time_ = std::filesystem::last_write_time(path, ec);
  note_ = std::format("saved {}", path);
  return true;
}

// A script changed by another editor is taken as an edit (hot reload), unless this one has unsaved edits.
void SceneUi::watch_file() {
  const auto now = Clock::now();
  if (path_.empty() || now - last_watch_ < std::chrono::milliseconds(500)) return;
  last_watch_ = now;
  std::error_code ec;
  const auto t = std::filesystem::last_write_time(path_, ec);
  if (ec || t == disk_time_) return;
  disk_time_ = t;
  if (modified_) {
    disk_changed_ = true;
    return;
  }
  if (const auto text = read_text(path_); text && *text != text_) {
    text_ = *text;
    session_.edit(text_, now);
    note_ = std::format("reloaded {} (changed on disk)", path_);
  }
}

bool SceneUi::replace_text(const std::string& from, const std::string& to) {
  const std::size_t at = text_.find(from);
  if (from.empty() || at == std::string::npos) return false;
  text_.replace(at, from.size(), to);
  modified_ = true;
  session_.edit(text_, Clock::now());
  goto_offset_ = static_cast<int>(at + to.size());  // the cursor and the view where the edit is, as after typing it
  scroll_line_ = 1 + static_cast<int>(std::count(text_.begin(), text_.begin() + static_cast<std::ptrdiff_t>(at), '\n'));
  scroll_wait_ = 3;
  return true;
}

bool SceneUi::picture_rect(ImVec2& min, ImVec2& max) const {
  min = pic_min_;
  max = pic_max_;
  return pic_drawn_;
}

// --- each frame -----------------------------------------------------------------------------------------------------

void SceneUi::frame(float x, float y, float w, float h, double dt) {
  watch_file();
  session_.update(Clock::now());
  if (playing_ && !scrubbing_) {
    session_.step(dt * static_cast<double>(speed_));
    if (loop_ && session_.scene() && session_.info().length > 0.f && session_.time() >= static_cast<double>(session_.info().length)) session_.restart();
  }
  // the probe and the overlay read the fields before the picture is drawn: overlapped, drawing frame f computes frame
  // f + 1, whose fields the reads would then see
  if (hover_) hover_probe_.ok = session_.probe(hover_px_.x, hover_px_.y, hover_probe_.x, hover_probe_.y, hover_probe_.f);
  if (pinned_) pin_probe_.ok = session_.probe(pin_px_.x, pin_px_.y, pin_probe_.x, pin_probe_.y, pin_probe_.f);
  if (overlay_field_ > 0) update_overlay();
  if (session_.draw()) {
    picture_.upload(session_.picture(), session_.info().width, session_.info().height);
    smooth(frame_ms_, session_.step_ms() + session_.draw_ms());
  }
  const float ew = std::clamp(w * 0.4f, 360.f, 720.f);
  script_window(x, y, ew, h);
  const float rx = x + ew + 6.f, rw = w - ew - 6.f;
  const float controls_h = std::clamp(h * 0.3f, 220.f, 320.f);
  scene_window(rx, y, rw, h - controls_h - 6.f);
  controls_window(rx, y + h - controls_h, rw, controls_h);
}

void SceneUi::script_window(float x, float y, float w, float h) {
  ImGui::SetNextWindowPos({x, y}, ImGuiCond_Always);
  ImGui::SetNextWindowSize({w, h}, ImGuiCond_Always);
  ImGui::Begin("Script", nullptr, kFixed);
  ImGui::SetNextItemWidth(std::max(80.f, ImGui::GetContentRegionAvail().x - 110.f));
  const bool enter = ImGui::InputTextWithHint("##path", "path/to/scene.nvfxs", &path_field_, ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  if (ImGui::Button("Open") || enter) open_file(path_field_);
  ImGui::SameLine();
  if (ImGui::Button("Save") || ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_S, ImGuiInputFlags_RouteGlobal)) save_file(path_field_);
  ImGui::TextDisabled("%s%s", path_.empty() ? "(no file)" : path_.c_str(), modified_ ? "  (unsaved)" : "");
  if (disk_changed_) {
    ImGui::TextColored(kBusyText, "changed on disk");
    ImGui::SameLine();
    if (ImGui::SmallButton("load it (drops the edits here)")) open_file(path_, session_.time());
  } else if (!note_.empty()) {
    ImGui::TextDisabled("%s", note_.c_str());
  }

  const auto& err = session_.error();
  const float status_h = err ? ImGui::GetTextLineHeightWithSpacing() * 4.f : ImGui::GetTextLineHeightWithSpacing() * 1.5f;
  editor(std::max(60.f, ImGui::GetContentRegionAvail().y - status_h));
  if (err) {
    ImGui::PushStyleColor(ImGuiCol_Text, kErrorText);
    ImGui::TextWrapped("%s", err->message.c_str());
    ImGui::PopStyleColor();
    if (err->line > 0 && ImGui::SmallButton("go to the error")) {
      const int line = std::min(err->line, static_cast<int>(line_starts_.size()));
      const int start = line_starts_[static_cast<std::size_t>(line - 1)];
      goto_offset_ = std::min(start + std::max(0, err->column - 1), static_cast<int>(text_.size()));
      scroll_line_ = line;
      scroll_wait_ = 3;
    }
    ImGui::SameLine();
    ImGui::TextDisabled(session_.has_scene() ? "the last good scene plays on" : "no scene yet");
  } else if (session_.building()) {
    ImGui::TextColored(kBusyText, "building...");
  } else if (session_.edit_pending()) {
    ImGui::TextDisabled("editing...");
  } else if (session_.has_scene()) {
    ImGui::TextColored(kGoodText, "built in %.0f ms", session_.build_ms());
  }
  if (!err && cursor_line_ >= 1 && cursor_line_ <= static_cast<int>(line_starts_.size())) {
    ImGui::SameLine();
    ImGui::TextDisabled("   line %d, column %d", cursor_line_, cursor_ - line_starts_[static_cast<std::size_t>(cursor_line_ - 1)] + 1);
  }
  ImGui::End();
}

// The text with line numbers. The text box is as tall as its lines, inside a child that scrolls, so the numbers and
// the text scroll together; the child follows the cursor.
void SceneUi::editor(float height) {
  line_starts_.clear();
  line_starts_.push_back(0);
  for (std::size_t i = 0; i < text_.size(); ++i)
    if (text_[i] == '\n') line_starts_.push_back(static_cast<int>(i + 1));
  const int lines = static_cast<int>(line_starts_.size());
  const ImGuiStyle& st = ImGui::GetStyle();
  ImGui::PushStyleColor(ImGuiCol_ChildBg, kEditorBg);
  ImGui::BeginChild("##editor", {0.f, height}, ImGuiChildFlags_Borders);
  ImGui::PopStyleColor();
  const float line_h = ImGui::GetTextLineHeight();
  const float gutter = ImGui::CalcTextSize("9999").x + st.ItemSpacing.x * 2.f;
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const float full_w = ImGui::GetContentRegionAvail().x;
  const float text_top = origin.y + st.FramePadding.y;
  const float scroll = ImGui::GetScrollY(), view_h = ImGui::GetWindowHeight();
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImGuiID id = ImGui::GetID("##text");

  const auto& err = session_.error();
  const int err_line = err && err->line > 0 && err->line <= lines ? err->line : 0;
  if (err_line > 0) {  // the line, and a mark under the column
    const float y0 = text_top + static_cast<float>(err_line - 1) * line_h;
    dl->AddRectFilled({origin.x, y0}, {origin.x + full_w, y0 + line_h}, kErrorLine);
    if (err->column > 0) {
      const char* ls = text_.c_str() + line_starts_[static_cast<std::size_t>(err_line - 1)];
      const char* le = ls;
      while (*le && *le != '\n' && le - ls < err->column - 1) ++le;
      const ImGuiInputTextState* state = ImGui::GetInputTextState(id);
      const float cx = origin.x + gutter + st.FramePadding.x + ImGui::CalcTextSize(ls, le).x - (state ? state->Scroll.x : 0.f);
      dl->AddLine({cx, y0 + line_h}, {cx + ImGui::CalcTextSize("mm").x, y0 + line_h}, kErrorMark, 2.f);
    }
  }
  const int first = std::max(0, static_cast<int>((scroll - st.FramePadding.y) / line_h));
  const int last = std::min(lines, first + static_cast<int>(view_h / line_h) + 2);
  char num[16];
  for (int i = first; i < last; ++i) {
    std::snprintf(num, sizeof num, "%d", i + 1);
    const float tw = ImGui::CalcTextSize(num).x;
    dl->AddText({origin.x + gutter - st.ItemSpacing.x - tw, text_top + static_cast<float>(i) * line_h},
                i + 1 == err_line ? kErrorMark : ImGui::GetColorU32(ImGuiCol_TextDisabled), num);
  }

  ImGui::SetCursorScreenPos({origin.x + gutter, origin.y});
  const float text_h = std::max(view_h - st.WindowPadding.y * 2.f, static_cast<float>(lines + 1) * line_h + st.FramePadding.y * 2.f);
  // a jump focuses the text box: ImGui drops such a (tab-like) focus on a box that takes Tab, so not while it waits
  const bool jumping = goto_offset_ >= 0;
  if (jumping && ImGui::GetActiveID() != id) ImGui::SetKeyboardFocusHere();
  ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
  const bool changed = ImGui::InputTextMultiline("##text", &text_, {full_w - gutter, text_h},
                                                 ImGuiInputTextFlags_CallbackAlways | (jumping ? 0 : ImGuiInputTextFlags_AllowTabInput), &SceneUi::callback, this);
  ImGui::PopStyleColor();
  if (changed) {
    modified_ = true;
    session_.edit(text_, Clock::now());
  }
  const auto line_top = [&](int line) { return st.WindowPadding.y + st.FramePadding.y + static_cast<float>(line - 1) * line_h; };
  // a jump (to the error, or to a test's edit): once the cursor is there, and a few frames after the text box was
  // focused, whose own scrolling (to show the box's top) would undo it
  if (scroll_line_ > 0 && goto_offset_ < 0 && --scroll_wait_ <= 0) {
    ImGui::SetScrollY(std::max(0.f, line_top(std::min(scroll_line_, lines)) - view_h / 3.f));
    scroll_line_ = 0;
  } else if (scroll_line_ == 0 && ImGui::GetActiveID() == id && cursor_line_ != last_cursor_line_) {  // follow the cursor
    const float ly = line_top(cursor_line_), bottom = scroll + view_h - st.WindowPadding.y - st.ScrollbarSize;
    if (ly < scroll) ImGui::SetScrollY(std::max(0.f, ly - st.WindowPadding.y));
    if (ly + line_h > bottom) ImGui::SetScrollY(ly + line_h - view_h + st.WindowPadding.y + st.ScrollbarSize);
  }
  last_cursor_line_ = cursor_line_;
  ImGui::EndChild();
}

int SceneUi::callback(ImGuiInputTextCallbackData* d) {
  auto* self = static_cast<SceneUi*>(d->UserData);
  if (self->goto_offset_ >= 0) {
    d->CursorPos = d->SelectionStart = d->SelectionEnd = std::min(self->goto_offset_, d->BufTextLen);
    self->goto_offset_ = -1;
  }
  self->cursor_ = d->CursorPos;
  int line = 1;
  for (int i = 0; i < d->CursorPos && i < d->BufTextLen; ++i)
    if (d->Buf[i] == '\n') ++line;
  self->cursor_line_ = line;
  return 0;
}

void SceneUi::scene_window(float x, float y, float w, float h) {
  ImGui::SetNextWindowPos({x, y}, ImGuiCond_Always);
  ImGui::SetNextWindowSize({w, h}, ImGuiCond_Always);
  ImGui::Begin("Scene", nullptr, kFixed | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  const nvfx_scene_info& in = session_.info();
  const float aspect = in.width > 0 ? static_cast<float>(in.height) / static_cast<float>(in.width) : 9.f / 16.f;
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const float below = ImGui::GetFrameHeightWithSpacing() + ImGui::GetTextLineHeightWithSpacing() * 2.f;
  float pw = avail.x, ph = pw * aspect;
  if (ph > avail.y - below) {
    ph = std::max(32.f, avail.y - below);
    pw = ph / aspect;
  }
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.f, (avail.x - pw) * 0.5f));
  const ImGuiIO& io = ImGui::GetIO();
  if (!picture_.empty()) {
    ImGui::Image(picture_.id(), {pw, ph});
    pic_min_ = ImGui::GetItemRectMin();
    pic_max_ = ImGui::GetItemRectMax();
    pic_drawn_ = true;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (overlay_field_ > 0 && !overlay_.empty()) dl->AddImage(overlay_.id(), pic_min_, pic_max_);
    const float sx = pw / static_cast<float>(std::max(1, in.width)), sy = ph / static_cast<float>(std::max(1, in.height));
    hover_ = ImGui::IsItemHovered();
    if (hover_) {
      hover_px_ = {(io.MousePos.x - pic_min_.x) / sx, (io.MousePos.y - pic_min_.y) / sy};
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        pinned_ = true;
        pin_px_ = hover_px_;
      }
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) pinned_ = false;
    }
    if (pinned_) {
      const ImVec2 c{pic_min_.x + pin_px_.x * sx, pic_min_.y + pin_px_.y * sy};
      dl->AddCircle(c, 7.f, IM_COL32(120, 220, 255, 255), 0, 1.5f);
      dl->AddLine({c.x - 11.f, c.y}, {c.x + 11.f, c.y}, IM_COL32(120, 220, 255, 255));
      dl->AddLine({c.x, c.y - 11.f}, {c.x, c.y + 11.f}, IM_COL32(120, 220, 255, 255));
    }
    if (!session_.scene()) dl->AddText({pic_min_.x + 8.f, pic_min_.y + 6.f}, IM_COL32(255, 220, 120, 255), "seeking...");
  } else {
    ImGui::Dummy({pw, ph});
    pic_drawn_ = false;
    const char* msg = session_.building() ? "building the scene..." : session_.error() ? "no scene: see the error under the script" : "open a scene script (.nvfxs)";
    const ImVec2 r0 = ImGui::GetItemRectMin(), ts = ImGui::CalcTextSize(msg);
    ImGui::GetWindowDrawList()->AddText({r0.x + (pw - ts.x) * 0.5f, r0.y + (ph - ts.y) * 0.5f}, ImGui::GetColorU32(ImGuiCol_TextDisabled), msg);
  }

  // transport
  if (ImGui::Button(playing_ ? "Pause" : "Play", {60.f, 0.f}) || (ImGui::IsKeyPressed(ImGuiKey_Space, false) && !io.WantTextInput)) playing_ = !playing_;
  ImGui::SameLine();
  if (ImGui::Button("Restart")) session_.restart();
  ImGui::SameLine();
  if (ImGui::Button("Step")) {
    playing_ = false;
    session_.step_frames(1);
  }
  ImGui::SameLine();
  ImGui::Checkbox("loop", &loop_);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(std::max(80.f, ImGui::GetContentRegionAvail().x - 190.f));
  float t = static_cast<float>(session_.time());
  if (ImGui::SliderFloat("##time", &t, 0.f, std::max(in.length, t), "%.2f s")) session_.seek(static_cast<double>(t));
  scrubbing_ = ImGui::IsItemActive();
  ImGui::SameLine();
  ImGui::SetNextItemWidth(130.f);
  ImGui::SliderFloat("speed", &speed_, 0.f, 4.f, "%.2fx");
  if (session_.has_scene()) {
    ImGui::Text("frame %d of %d at %.0f fps, %dx%d, %d modules, %d rules; %.1f ms a frame (state and picture, %d threads%s)", session_.frame(), in.frames,
                static_cast<double>(in.fps), in.width, in.height, in.n_modules, in.n_rules, frame_ms_, in.threads, in.overlap ? ", overlapped" : "");
  } else {
    ImGui::TextDisabled("no scene");
  }
  if (session_.building() || session_.seeking()) {
    ImGui::SameLine();
    ImGui::TextColored(kBusyText, session_.building() ? "  building..." : "  seeking...");
  }
  ImGui::End();
}

void SceneUi::controls_window(float x, float y, float w, float h) {
  ImGui::SetNextWindowPos({x, y}, ImGuiCond_Always);
  ImGui::SetNextWindowSize({w, h}, ImGuiCond_Always);
  ImGui::Begin("Scene controls", nullptr, kFixed);
  if (!ImGui::BeginTable("##controls", 4, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp)) {
    ImGui::End();
    return;
  }
  ImGui::TableSetupColumn("inputs", ImGuiTableColumnFlags_WidthStretch, 1.15f);
  ImGui::TableSetupColumn("rules", ImGuiTableColumnFlags_WidthStretch, 1.f);
  ImGui::TableSetupColumn("probe", ImGuiTableColumnFlags_WidthStretch, 0.9f);
  ImGui::TableSetupColumn("effects", ImGuiTableColumnFlags_WidthStretch, 1.f);
  nvfx_scene* s = session_.scene();

  ImGui::TableNextColumn();
  ImGui::SeparatorText("inputs");
  if (session_.inputs().empty()) ImGui::TextDisabled("none (input NAME = value)");
  for (InputValue& in : session_.inputs()) {
    ImGui::PushID(in.name.c_str());
    float lo = 0.f, hi = 1.f;
    input_range(in.script, lo, hi);
    ImGui::SetNextItemWidth(std::max(60.f, ImGui::GetContentRegionAvail().x * 0.55f));
    ImGui::SliderFloat(in.name.c_str(), &in.value, lo, hi);
    if (in.value != in.script) {
      ImGui::SameLine();
      if (ImGui::SmallButton("reset")) in.value = in.script;
    }
    ImGui::PopID();
  }

  ImGui::TableNextColumn();
  ImGui::SeparatorText("rules");
  if (session_.rules().empty()) ImGui::TextDisabled("no named rules (when ... as NAME:)");
  for (const std::string& r : session_.rules()) {
    if (ImGui::Button(r.c_str()) && !session_.trigger(r)) trigger_note_ = std::format("'{}' cannot be fired from here (a landing rule)", r);
    int count = 0;
    float last = 0.f;
    if (s && nvfx_scene_rule_state(s, r.c_str(), &count, &last) == NVFX_OK) {
      ImGui::SameLine();
      if (count > 0) {
        ImGui::TextDisabled("%dx, last %.2f s", count, static_cast<double>(last));
      } else {
        ImGui::TextDisabled("not yet");
      }
    }
  }
  if (!trigger_note_.empty()) ImGui::TextWrapped("%s", trigger_note_.c_str());

  ImGui::TableNextColumn();
  ImGui::SeparatorText("field probe");
  const auto show = [](const char* what, const Probe& p) {
    if (!p.ok) return;
    ImGui::Text("%s (%.0f, %.0f)", what, static_cast<double>(p.x), static_cast<double>(p.y));
    ImGui::Text("  heat %.3f  soot %.3f", static_cast<double>(p.f.heat), static_cast<double>(p.f.soot));
    ImGui::Text("  velocity (%.0f, %.0f) px/s", static_cast<double>(p.f.u), static_cast<double>(p.f.v));
  };
  if (hover_) show("pointer", hover_probe_);
  if (pinned_) show("pinned", pin_probe_);
  if (!hover_ && !pinned_) ImGui::TextDisabled("point at the picture; click pins, right-click unpins");
  const char* fields[] = {"no overlay", "heat overlay", "soot overlay"};
  ImGui::SetNextItemWidth(std::max(60.f, ImGui::GetContentRegionAvail().x * 0.8f));
  ImGui::Combo("##overlay", &overlay_field_, fields, 3);

  ImGui::TableNextColumn();
  ImGui::SeparatorText("effects");
  SessionSettings set = session_.settings();
  bool apply = false;
  ImGui::SetNextItemWidth(std::max(60.f, ImGui::GetContentRegionAvail().x - 50.f));
  if (ImGui::InputTextWithHint("##dir", "effects folder", &dir_field_, ImGuiInputTextFlags_EnterReturnsTrue)) apply = true;
  if (ImGui::IsItemHovered() && !dir_field_.empty()) ImGui::SetTooltip("%s", dir_field_.c_str());
  ImGui::SameLine();
  if (ImGui::Button("use")) apply = true;
  if (apply) set.effects_dir = dir_field_;
  int si = static_cast<int>(set.stand_ins);
  const char* stand[] = {"no stand-ins", "stand-ins for missing", "stand-ins for all"};
  ImGui::SetNextItemWidth(std::max(60.f, ImGui::GetContentRegionAvail().x * 0.8f));
  if (ImGui::Combo("##standins", &si, stand, 3)) {
    set.stand_ins = static_cast<StandIns>(si);
    apply = true;
  }
  int threads = set.threads;
  ImGui::SetNextItemWidth(80.f);
  ImGui::SliderInt("threads", &threads, 1, 8);
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    set.threads = threads;
    apply = true;
  }
  bool overlap = set.overlap != 0;
  if (ImGui::Checkbox("overlap", &overlap)) {
    set.overlap = overlap ? -1 : 0;
    apply = true;
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("draw each picture while the next frame's state is computed (needs 2 threads)");
  ImGui::SameLine();
  if (ImGui::Checkbox("from 0", &set.restart_on_reload)) apply = true;
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("a rebuilt scene starts at 0 (else it is played to the time of the edit)");
  if (apply) session_.set_settings(set);
  if (ImGui::Button("reload effects")) session_.reload_effects();
  for (const EffectUse& e : session_.effects()) {
    const char* from = e.from == EffectUse::From::file ? "" : e.from == EffectUse::From::stand_in ? " (stand-in)" : " (missing)";
    ImGui::TextDisabled("%s%s", e.file.c_str(), from);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("effect %s: %s", e.name.c_str(), e.from == EffectUse::From::file       ? e.path.c_str()
                                                         : e.from == EffectUse::From::stand_in ? "an untrained stand-in (its file is not used)"
                                                                                               : "not in the folder");
    }
  }
  ImGui::EndTable();
  ImGui::End();
}

// The bus under the picture, one value per 8 pixels, as a translucent layer: where the scene's heat or soot is.
void SceneUi::update_overlay() {
  nvfx_scene* s = session_.scene();
  if (!s) return;
  const nvfx_scene_info& in = session_.info();
  constexpr int kCell = 8;
  const int ow = std::max(1, in.scene_width / kCell), oh = std::max(1, in.scene_height / kCell);
  if (ow != overlay_w_ || oh != overlay_h_) {
    overlay_w_ = ow;
    overlay_h_ = oh;
    overlay_values_.assign(static_cast<std::size_t>(ow) * static_cast<std::size_t>(oh), 0.f);
    overlay_rgba_.assign(overlay_values_.size() * 4, 0);
  }
  float cx = 0.f, cy = 0.f;
  nvfx_scene_camera(s, &cx, &cy);
  const nvfx_scene_field field = overlay_field_ == 1 ? NVFX_FIELD_HEAT : NVFX_FIELD_SOOT;
  const float cell = static_cast<float>(kCell);
  nvfx_scene_sample_grid(s, field, cx + 0.5f * cell, cy + 0.5f * cell, cell, cell, ow, oh, overlay_values_.data(), static_cast<std::size_t>(ow));
  for (std::size_t i = 0; i < overlay_values_.size(); ++i) {
    const float a = std::sqrt(std::clamp(overlay_values_[i], 0.f, 1.f));  // 1 shows fully; the root shows faint values too
    std::uint8_t* p = overlay_rgba_.data() + i * 4;
    p[0] = overlay_field_ == 1 ? 255 : 80;
    p[1] = overlay_field_ == 1 ? 120 : 200;
    p[2] = overlay_field_ == 1 ? 30 : 255;
    p[3] = static_cast<std::uint8_t>(a * 190.f);
  }
  overlay_.upload(overlay_rgba_, ow, oh);
}

}  // namespace nfx::viewer
