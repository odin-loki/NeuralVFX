// A tiny command-line parser shared by the tools: --name value pairs, --flag switches and positional words.
#pragma once

#include <cstdlib>
#include <print>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace nfx::tools {

class Args {
 public:
  Args(int argc, char** argv, std::set<std::string> flags) : flags_(std::move(flags)) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) == 0) {
        const std::string key = a.substr(2);
        if (flags_.count(key)) {
          set_.insert(key);
        } else if (i + 1 < argc) {
          values_[key] = argv[++i];
        } else {
          throw std::invalid_argument("--" + key + " needs a value");
        }
      } else {
        positional_.push_back(a);
      }
    }
  }
  bool flag(const std::string& k) const { return set_.count(k) > 0; }
  bool has(const std::string& k) const { return values_.count(k) > 0; }
  std::string str(const std::string& k, const std::string& def = "") const {
    used_.insert(k);
    const auto it = values_.find(k);
    return it == values_.end() ? def : it->second;
  }
  std::string need(const std::string& k) const {
    if (!has(k)) throw std::invalid_argument("missing --" + k);
    return str(k);
  }
  int i(const std::string& k, int def) const { return has(k) ? std::stoi(str(k)) : def; }
  float f(const std::string& k, float def) const { return has(k) ? std::stof(str(k)) : def; }
  std::uint64_t u64(const std::string& k, std::uint64_t def) const { return has(k) ? std::stoull(str(k)) : def; }
  const std::vector<std::string>& positional() const { return positional_; }
  // Values that no code asked for: usually a typo.
  std::vector<std::string> unused() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : values_) {
      if (!used_.count(k)) out.push_back(k);
    }
    return out;
  }
  void warn_unused() const {
    for (const auto& k : unused()) std::println(stderr, "warning: unused option --{}", k);
  }

 private:
  std::set<std::string> flags_, set_;
  std::map<std::string, std::string> values_;
  std::vector<std::string> positional_;
  mutable std::set<std::string> used_;
};

inline std::vector<float> parse_floats(const std::string& s) {
  std::vector<float> v;
  std::size_t p = 0;
  while (p < s.size()) {
    const std::size_t q = s.find(',', p);
    v.push_back(std::stof(s.substr(p, q == std::string::npos ? std::string::npos : q - p)));
    if (q == std::string::npos) break;
    p = q + 1;
  }
  return v;
}

}  // namespace nfx::tools
