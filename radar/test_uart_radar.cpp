// test_uart_radar.cpp - checks that the C4002 radar is talking to the Jetson over UART.
//
// A Linux port of DFRobot's Arduino example (DFRobot_C4002.h needs Arduino.h and
// HardwareSerial, so it can't be used here). It talks to /dev/ttyTHS1 directly with termios.
//
// Wiring (Jetson 40-pin header -> C4002)
//   pin 6 GND -> GND,  pin 2 5V -> VIN,  pin 8 TX -> RX,  pin 10 RX -> TX
//   Pins 8/10 are /dev/ttyTHS1, 115200 baud, 8N1.
//
// What it checks
//   1. RX: waits for valid C4002 frames (header FA F5 AA A5, good checksum) and prints the
//      telemetry, like the Arduino loop() does.
//   2. TX: sends a read-only "get version" request and waits for the sensor's reply.
//
// Build and run
//   g++ -O2 -std=c++17 -o test_uart_radar test_uart_radar.cpp
//   ./test_uart_radar                         # listen 10 s on /dev/ttyTHS1
//   ./test_uart_radar --seconds 30 --port /dev/ttyTHS1
//   ./test_uart_radar --period 10             # also set report period to 1.0 s (writes sensor config)
//   ./test_uart_radar --no-tx                 # only listen, send nothing
//
// Stop anything else using the port first (e.g. the radar_publisher node).

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Bytes = std::vector<uint8_t>;
using Clock = std::chrono::steady_clock;

static const uint8_t HEADER[4] = {0xFA, 0xF5, 0xAA, 0xA5};
static const uint8_t TYPE_WRITE_REQ = 0x00, TYPE_READ_REQ = 0x01, TYPE_WRITE_RESP = 0x02,
                     TYPE_READ_RESP = 0x03, TYPE_NOTE = 0x04;
static const uint8_t CMD_GET_VERSION = 0x82, CMD_SET_REPORT_PERIOD = 0x83, NOTE_RESULT = 0x60,
                     NOTE_CALIBRATION = 0x03;

// ---------- serial port ----------

static int open_port(const char *path, int baud) {
    speed_t speed;
    switch (baud) {
        case 9600: speed = B9600; break;
        case 57600: speed = B57600; break;
        case 115200: speed = B115200; break;
        case 230400: speed = B230400; break;
        case 460800: speed = B460800; break;
        case 921600: speed = B921600; break;
        default: std::fprintf(stderr, "unsupported baud %d\n", baud); return -1;
    }
    int fd = ::open(path, O_RDWR | O_NOCTTY);
    if (fd < 0) return -1;
    termios tio{};
    if (tcgetattr(fd, &tio) != 0) { ::close(fd); return -1; }
    cfmakeraw(&tio);                       // 8 data bits, no parity, no echo, no line editing
    tio.c_cflag &= ~(CSTOPB | CRTSCTS);    // 1 stop bit, no hardware flow control
    tio.c_cflag |= CLOCAL | CREAD;
    tio.c_cc[VMIN] = 0;                    // read() returns after 100 ms even with no data
    tio.c_cc[VTIME] = 1;
    cfsetispeed(&tio, speed);
    cfsetospeed(&tio, speed);
    if (tcsetattr(fd, TCSANOW, &tio) != 0) { ::close(fd); return -1; }
    tcflush(fd, TCIOFLUSH);
    return fd;
}

// ---------- framing ----------

static bool checksum_ok(const Bytes &f) {
    if (f.size() < 8) return false;
    uint32_t sum = 0;
    for (size_t i = 0; i + 2 < f.size(); ++i) sum += f[i];
    return (sum & 0xFFFF) == static_cast<uint32_t>(f[f.size() - 2] | (f[f.size() - 1] << 8));
}

static Bytes build_frame(uint8_t type, uint8_t cmd, const Bytes &params) {
    // header, total length, 0x00, type, cmd, 0x00, inner length, params, checksum
    uint16_t inner = static_cast<uint16_t>(4 + params.size());
    uint16_t total = static_cast<uint16_t>(inner + 10);
    Bytes f(HEADER, HEADER + 4);
    f.insert(f.end(), {static_cast<uint8_t>(total & 0xFF), static_cast<uint8_t>(total >> 8), 0x00, type,
                       cmd, 0x00, static_cast<uint8_t>(inner & 0xFF), static_cast<uint8_t>(inner >> 8)});
    f.insert(f.end(), params.begin(), params.end());
    uint32_t sum = 0;
    for (uint8_t b : f) sum += b;
    f.push_back(sum & 0xFF);
    f.push_back((sum >> 8) & 0xFF);
    return f;
}

// Pulls complete frames out of buf; anything before a header counts as junk.
struct Framer {
    Bytes buf;
    size_t junk = 0;

    std::vector<Bytes> feed(const uint8_t *data, size_t n) {
        buf.insert(buf.end(), data, data + n);
        std::vector<Bytes> out;
        for (;;) {
            size_t i = 0;
            while (i + 4 <= buf.size() && std::memcmp(&buf[i], HEADER, 4) != 0) ++i;
            if (i + 4 > buf.size()) {          // no header: keep the last 3 bytes in case one is split
                size_t keep = buf.size() < 3 ? buf.size() : 3;
                junk += buf.size() - keep;
                buf.erase(buf.begin(), buf.end() - keep);
                return out;
            }
            junk += i;
            buf.erase(buf.begin(), buf.begin() + i);
            if (buf.size() < 6) return out;
            size_t total = buf[4] | (buf[5] << 8);
            if (total < 10 || total > 256) {   // not a real frame, skip this header
                buf.erase(buf.begin());
                ++junk;
                continue;
            }
            if (buf.size() < total) return out;
            out.emplace_back(buf.begin(), buf.begin() + total);
            buf.erase(buf.begin(), buf.begin() + total);
        }
    }
};

static std::string hex(const Bytes &b, size_t max = 64) {
    std::string s;
    char tmp[4];
    for (size_t i = 0; i < b.size() && i < max; ++i) {
        std::snprintf(tmp, sizeof tmp, "%02x ", b[i]);
        s += tmp;
    }
    return s;
}

// ---------- telemetry (same fields as the Arduino example) ----------

static const char *state_name(uint8_t s) {
    switch (s) {
        case 0: return "No target";
        case 1: return "Static presence";
        case 2: return "Motion";
        default: return "Unknown";
    }
}

static const char *direction_name(uint8_t d) {
    switch (d) {
        case 0: return "away";
        case 1: return "none";
        case 2: return "approaching";
        default: return "?";
    }
}

static void print_frame(const Bytes &f) {
    uint8_t type = f[7], cmd = f[8];
    if (type == TYPE_NOTE && cmd == NOTE_RESULT && f.size() >= 12 + 18 + 2) {
        const uint8_t *p = &f[12];
        uint8_t state = p[0];
        double light_lux = (p[1] | (p[2] << 8)) * 0.1;
        double pres_dist_m = (p[9] | (p[10] << 8)) * 0.01;
        uint8_t pres_energy = p[11];
        double mot_dist_m = (p[12] | (p[13] << 8)) * 0.01;
        double mot_speed = static_cast<int16_t>(p[14] | (p[15] << 8)) * 0.01;
        uint8_t mot_energy = p[16];
        std::printf("  State: %-16s presence %.2f m (E%u)  motion %.2f m %+.2f m/s %s (E%u)  light %.1f lx\n",
                    state_name(state), pres_dist_m, pres_energy, mot_dist_m, mot_speed,
                    direction_name(p[17]), mot_energy, light_lux);
    } else if (type == TYPE_NOTE && cmd == NOTE_CALIBRATION && f.size() >= 16) {
        std::printf("  Calibrating, %u s left\n", f[12] | (f[13] << 8));
    } else {
        std::printf("  frame type 0x%02X cmd 0x%02X: %s\n", type, cmd, hex(f).c_str());
    }
}

// ---------- main ----------

struct Stats {
    size_t bytes = 0, frames = 0, good = 0;
    Bytes first_raw;
};

// Reads for `seconds`, printing good frames. Returns the good frames seen.
static std::vector<Bytes> listen(int fd, double seconds, Framer &fr, Stats &st, bool verbose) {
    std::vector<Bytes> good;
    auto end = Clock::now() + std::chrono::duration<double>(seconds);
    uint8_t chunk[256];
    while (Clock::now() < end) {
        ssize_t n = ::read(fd, chunk, sizeof chunk);
        if (n <= 0) continue;
        st.bytes += n;
        for (ssize_t i = 0; i < n && st.first_raw.size() < 64; ++i) st.first_raw.push_back(chunk[i]);
        for (auto &f : fr.feed(chunk, n)) {
            ++st.frames;
            if (!checksum_ok(f)) {
                if (verbose) std::printf("  [BAD checksum] %s\n", hex(f).c_str());
                continue;
            }
            ++st.good;
            if (verbose) print_frame(f);
            good.push_back(std::move(f));
        }
    }
    return good;
}

int main(int argc, char **argv) {
    std::string port = "/dev/ttyTHS1";
    int baud = 115200;
    double seconds = 10.0;
    int period = -1;   // -1 = don't change the report period
    bool do_tx = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *name) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--port") port = next("--port");
        else if (a == "--baud") baud = std::atoi(next("--baud"));
        else if (a == "--seconds") seconds = std::atof(next("--seconds"));
        else if (a == "--period") period = std::atoi(next("--period"));
        else if (a == "--no-tx") do_tx = false;
        else {
            std::printf("usage: %s [--port P] [--baud B] [--seconds S] [--period N(x0.1s)] [--no-tx]\n", argv[0]);
            return a == "-h" || a == "--help" ? 0 : 2;
        }
    }

    int fd = open_port(port.c_str(), baud);
    if (fd < 0) {
        std::printf("FAIL: can't open %s: %s\n", port.c_str(), std::strerror(errno));
        std::printf("      Is something else using it (radar_publisher)? Are you in the 'dialout' group?\n");
        return 2;
    }

    Framer fr;
    Stats st;

    // ---- 1. RX: like c4002.begin() + loop() in the Arduino example ----
    std::printf("[1/2] Listening on %s @ %d for %.0f s ...\n", port.c_str(), baud, seconds);
    listen(fd, seconds, fr, st, true);

    std::printf("\n  bytes received : %zu\n", st.bytes);
    std::printf("  frames found   : %zu (%zu with good checksum)\n", st.frames, st.good);
    std::printf("  unframed bytes : %zu\n", fr.junk);
    if (!st.first_raw.empty()) std::printf("  first bytes    : %s\n", hex(st.first_raw).c_str());

    const char *rx = "PASS";
    if (st.bytes == 0) {
        rx = "FAIL";
        std::printf("\n  RX FAIL: no bytes at all. Check 5 V on VIN (blue RUN LED on), shared GND, and that\n"
                    "  sensor TX goes to Jetson pin 10 (RX). TX/RX may be swapped.\n");
    } else if (st.good == 0) {
        rx = "FAIL";
        std::printf("\n  RX FAIL: bytes arrive but no valid frames. Usually wrong baud, a missing GND, or a\n"
                    "  floating/noisy RX line (sensor TX not really on pin 10).\n");
    } else {
        std::printf("\n  RX PASS: %.1f valid frames/s from the sensor.\n", st.good / seconds);
    }

    // ---- 2. TX: does the sensor answer us? ----
    const char *tx = "SKIPPED";
    if (do_tx) {
        Bytes req = build_frame(TYPE_READ_REQ, CMD_GET_VERSION, {});
        std::printf("\n[2/2] Sending 'get version' request: %s\n", hex(req).c_str());
        tcflush(fd, TCIFLUSH);
        if (::write(fd, req.data(), req.size()) != static_cast<ssize_t>(req.size())) perror("write");
        tcdrain(fd);
        Stats st2;
        Framer fr2;
        tx = "NO REPLY";
        for (auto &f : listen(fd, 2.0, fr2, st2, false)) {
            if ((f[7] == TYPE_READ_RESP || f[7] == TYPE_WRITE_RESP) && f[8] == CMD_GET_VERSION) {
                tx = "PASS";
                std::printf("  reply: %s\n", hex(f).c_str());
                std::string ver;
                for (size_t i = 12; i + 2 < f.size(); ++i)
                    if (f[i] >= 32 && f[i] < 127) ver += static_cast<char>(f[i]);
                if (!ver.empty()) std::printf("  version: %s\n", ver.c_str());
                std::printf("  TX PASS: the sensor answered, so Jetson pin 8 (TX) -> sensor RX works.\n");
                break;
            }
        }
        if (std::strcmp(tx, "PASS") != 0) {
            std::printf("  No reply to the version request.\n");
            if (std::strcmp(rx, "PASS") == 0)
                std::printf("  RX works, so either pin 8 isn't reaching the sensor's RX or the request format\n"
                            "  is off (taken from the community c4002 library). Treat as inconclusive.\n");
        }

        // Optional: same as c4002.setReportPeriod(N) in the Arduino example. Writes sensor config.
        if (period > 0) {
            Bytes set = build_frame(TYPE_WRITE_REQ, CMD_SET_REPORT_PERIOD, {static_cast<uint8_t>(period & 0xFF)});
            std::printf("\n  Setting report period to %.1f s: %s\n", period / 10.0, hex(set).c_str());
            if (::write(fd, set.data(), set.size()) != static_cast<ssize_t>(set.size())) perror("write");
            tcdrain(fd);
            Stats st3;
            Framer fr3;
            bool acked = false;
            for (auto &f : listen(fd, 2.0, fr3, st3, false))
                if (f[7] == TYPE_WRITE_RESP && f[8] == CMD_SET_REPORT_PERIOD) {
                    acked = true;
                    std::printf("  ack: %s (status 0x%02X, 0x01 = ok)\n", hex(f).c_str(), f[9]);
                }
            if (!acked) std::printf("  no ack for set report period\n");
        }
    }

    ::close(fd);
    std::printf("\nRESULT: RX %s, TX %s\n", rx, tx);
    return std::strcmp(rx, "PASS") == 0 ? 0 : 1;
}
