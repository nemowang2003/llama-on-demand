{pkgs}: final: prev: {
  # The locked generic wheel loads libsndfile with ctypes, so autoPatchelf
  # cannot discover this dependency. Keep the Python wheel/version unchanged.
  soundfile = prev.soundfile.overrideAttrs (old: {
    postInstall =
      (old.postInstall or "")
      + ''
        substituteInPlace "$out/${final.python.sitePackages}/soundfile.py" \
          --replace-fail "_find_library('sndfile')" "'${pkgs.lib.getLib pkgs.libsndfile}/lib/libsndfile${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}'"
      '';
  });
}
