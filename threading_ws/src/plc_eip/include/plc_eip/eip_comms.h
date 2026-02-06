
#include <cipster_api.h>
#include <string.h>
#include <stdlib.h>

#define DEMO_APP_INPUT_ASSEMBLY_NUM 100                 // 0x064
#define DEMO_APP_OUTPUT_ASSEMBLY_NUM 150                // 0x096
#define DEMO_APP_CONFIG_ASSEMBLY_NUM 151                // 0x097
#define DEMO_APP_HEARTBEAT_INPUT_ONLY_ASSEMBLY_NUM 152  // 0x098
#define DEMO_APP_HEARTBEAT_LISTEN_ONLY_ASSEMBLY_NUM 153 // 0x099
#define DEMO_APP_EXPLICT_ASSEMBLY_NUM 154               // 0x09A

bool parse_mac(const char *mac_str, uint8_t mac_out[6]);

/** @brief Signal handler function for ending stack execution
 *
 * @param signal the signal we received
 */
void LeaveStack(int signal);

volatile bool g_end_stack;

class eip_comms
{
public:
    eip_comms(
        char *ip_addr,
        char *ip_mask,
        char *ip_gateway,
        char *domain,
        char *host_addr,
        char *mac_addr);

    ~eip_comms();

private:
    uint8_t g_assembly_data064[128]; // Input
    uint8_t g_assembly_data096[128]; // Output
    uint8_t g_assembly_data097[64];  // Config
    uint8_t g_assembly_data09A[128]; // Explicit

};