// An RGBA8 OpenGL texture for ImGui::Image (nearest or linear filtering), reallocated only when its size changes.
#pragma once

#include <imgui.h>

#include <GLFW/glfw3.h>

#include <cstdint>
#include <span>

namespace nfx::viewer {

class Texture {
 public:
  explicit Texture(bool linear = false) : linear_(linear) { glGenTextures(1, &id_); }
  ~Texture() { glDeleteTextures(1, &id_); }
  Texture(const Texture&) = delete;
  Texture& operator=(const Texture&) = delete;
  void upload(std::span<const std::uint8_t> rgba, int width, int height) {
    glBindTexture(GL_TEXTURE_2D, id_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (width == w_ && height == h_) {
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
      return;
    }
    const GLint filter = linear_ ? GL_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    w_ = width;
    h_ = height;
  }
  void upload(std::span<const std::uint8_t> rgba, int size) { upload(rgba, size, size); }
  ImTextureID id() const { return static_cast<ImTextureID>(id_); }
  bool empty() const { return w_ == 0; }

 private:
  GLuint id_ = 0;
  int w_ = 0, h_ = 0;
  bool linear_ = false;
};

}  // namespace nfx::viewer
