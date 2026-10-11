#!/usr/bin/env python3
"""Adds --expert-profile-save PATH and --expert-profile-save-every SEC to gptoss_ext / mistral4_ext IN PLACE, on the source you have.

  python3 tools/add_profile_save.py            # run from the project root (or pass the root as argument)

What it adds (idempotent; nothing is written if any anchor is missing):
  * RunOpts: expert_profile_save, expert_profile_save_every
  * serve loop: the profile is written to --expert-profile-save (default: the --expert-profile path)
      - every SEC seconds while generating (0 = off), after EVERY request, and on QUIT / end of input  -> message "expert profile saved to ..."
  * one-shot mode (--prompt-ids): written at exit
  * the new flags in the runner's option parser and usage text
Then rebuild:  sh build.sh --cuda   (this tree is built with build.sh, not CMake)."""
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent
hdrs = list(root.glob("include/*/model.hpp"))
serve = root / "src" / "serve_loop.cpp"
mains = list((root / "src").glob("*_run_main.cpp"))
if len(hdrs) != 1 or not serve.exists() or len(mains) != 1:
    sys.exit("not a gptoss_ext/mistral4_ext root: %s (need include/*/model.hpp, src/serve_loop.cpp, src/*_run_main.cpp)" % root)
hdr, main = hdrs[0], mains[0]
H, S, M = hdr.read_text(), serve.read_text(), main.read_text()
if "expert_profile_save" in H and "save_prof(" in S and "--expert-profile-save" in M:
    sys.exit("already applied")
prog = "gptoss_run" if "gptoss_run:" in S else "m4_run"
missing = []


def sub(text, pattern, repl, what, flags=0, count=1):
    new, n = re.subn(pattern, repl, text, count=count, flags=flags)
    if n == 0:
        missing.append(what)
        return text
    return new


# ---- RunOpts
if "expert_profile_save" not in H:
    H = sub(H, r"(\n[ \t]*std::string expert_profile;[^\n]*\n)",
            r"\1    std::string expert_profile_save;   // where the profile is written (default: the --expert-profile path)\n"
            r"    double expert_profile_save_every = 0;   // seconds between periodic saves while generating (0 = only after each request and at exit)\n",
            "RunOpts: line 'std::string expert_profile;' in " + hdr.name)

# ---- serve loop
if "save_prof(" not in S:
    S = sub(S, r"(\n[ \t]*LineReader in;\n)", r"""\1    const std::string prof_path = !o.expert_profile_save.empty() ? o.expert_profile_save : o.expert_profile;
    auto last_prof = std::chrono::steady_clock::now();
    auto save_prof = [&](const char* why) {              // every path that writes the profile goes through here
        if (prof_path.empty()) return;
        std::string perr;
        if (m.save_profile(prof_path, perr)) std::fprintf(stderr, "%s: expert profile saved to %s (%s)\\n", "PROG", prof_path.c_str(), why);
        else std::fprintf(stderr, "%s: profile save FAILED: %s\\n", "PROG", perr.c_str());
        std::fflush(stderr);
        last_prof = std::chrono::steady_clock::now();
    };
""".replace("PROG", prog), "serve_loop: 'LineReader in;'")
    S = sub(S, r"(\n[ \t]*m\.forward\(tok, pos, &logits\);\n[ \t]*\+\+pos;\n)",
            r"""\1            if (o.expert_profile_save_every > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - last_prof).count() >= o.expert_profile_save_every)
                save_prof("periodic");
""", "serve_loop: generation step 'm.forward(tok, pos, &logits); ++pos;'")
    S = sub(S, r"[ \t]*if \(!o\.expert_profile\.empty\(\)\) \{[^\n]*\n[ \t]*std::string perr;\n[ \t]*if \(!m\.save_profile\([^\n]*\n[ \t]*\}\n",
            '        save_prof("request");\n', "serve_loop: the old per-request profile block")
    i = S.rfind("\n    return 0;\n}\n\n}  // namespace")
    if i < 0:
        missing.append("serve_loop: final 'return 0;' of serve_loop()")
    else:
        S = S[:i] + '\n    save_prof("exit");' + S[i:]

# ---- runner
if "--expert-profile-save" not in M:
    M = sub(M, r'([ \t]*else if \(a == "--expert-profile"\) o\.expert_profile = val\(\);\n)',
            r'''\1        else if (a == "--expert-profile-save") o.expert_profile_save = val();
        else if (a == "--expert-profile-save-every") o.expert_profile_save_every = std::atof(val());
''', "runner: the '--expert-profile' option")
    M = sub(M, r"(\n        m\.forward\(tok, pos\+\+, &lg\);\n    \}\n)(    return 0;)", r"""\1    {   // one-shot mode: write the profile at exit
        const std::string pp = !o.expert_profile_save.empty() ? o.expert_profile_save : o.expert_profile;
        if (!pp.empty()) {
            std::string perr;
            if (m.save_profile(pp, perr)) std::fprintf(stderr, "PROG: expert profile saved to %s (exit)\\n", pp.c_str());
            else std::fprintf(stderr, "PROG: profile save FAILED: %s\\n", perr.c_str());
        }
    }
\2""".replace("PROG", prog), "runner: end of the one-shot generation loop")
    if "[--expert-profile-save" not in M:
        M = M.replace("[--expert-profile FILE]", "[--expert-profile FILE] [--expert-profile-save FILE] [--expert-profile-save-every SEC]", 1)

if missing:
    sys.exit("NOTHING WRITTEN. Anchors not found:\n  - " + "\n  - ".join(missing) + "\nSend me these functions as they are in your tree.")
hdr.write_text(H); serve.write_text(S); main.write_text(M)
print("ok: %s, %s, %s patched. Rebuild: sh build.sh --cuda" % (hdr.name, serve.name, main.name))
