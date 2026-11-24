use std::{mem, slice};

use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, NO_ERROR};
use windows_sys::Win32::NetworkManagement::IpHelper::{
    GetExtendedTcpTable, MIB_TCPROW_OWNER_PID, MIB_TCPTABLE_OWNER_PID, MIB_TCP_STATE_LISTEN,
    TCP_TABLE_OWNER_PID_LISTENER,
};
use windows_sys::Win32::Networking::WinSock::AF_INET;

/// Require the exact IPv4 loopback listener for a child already held by the desktop.
pub(crate) fn owns_loopback_listener(port: u16, pid: u32) -> bool {
    let mut buffer = vec![0u32; 1024];
    for _ in 0..3 {
        let mut size = (buffer.len() * mem::size_of::<u32>()) as u32;
        // SAFETY: The initialized, word-aligned buffer is writable for `size` bytes.
        let result = unsafe {
            GetExtendedTcpTable(
                buffer.as_mut_ptr().cast(),
                &mut size,
                0,
                AF_INET as u32,
                TCP_TABLE_OWNER_PID_LISTENER,
                0,
            )
        };
        match result {
            NO_ERROR => return table_matches(&buffer, size as usize, port, pid),
            ERROR_INSUFFICIENT_BUFFER if size <= 16 * 1024 * 1024 => {
                buffer.resize((size as usize).div_ceil(mem::size_of::<u32>()), 0);
            }
            _ => return false,
        }
    }
    false
}

fn table_matches(buffer: &[u32], size: usize, port: u16, pid: u32) -> bool {
    let rows_offset = mem::offset_of!(MIB_TCPTABLE_OWNER_PID, table);
    if size < rows_offset || size > mem::size_of_val(buffer) {
        return false;
    }
    let count = buffer[0] as usize;
    if count > (size - rows_offset) / mem::size_of::<MIB_TCPROW_OWNER_PID>() {
        return false;
    }
    // SAFETY: The checked rows fit in the buffer and the C layout aligns them on a word
    // boundary. Each row consists of six initialized u32 values.
    let rows = unsafe {
        let first_row = buffer.as_ptr().cast::<u8>().add(rows_offset).cast();
        slice::from_raw_parts::<MIB_TCPROW_OWNER_PID>(first_row, count)
    };
    let loopback = u32::from_ne_bytes([127, 0, 0, 1]);
    let mut found_loopback = false;
    for row in rows {
        if row.dwState != MIB_TCP_STATE_LISTEN as u32
            || u16::from_be(row.dwLocalPort as u16) != port
        {
            continue;
        }
        if row.dwLocalAddr == loopback || row.dwLocalAddr == 0 {
            if row.dwOwningPid != pid {
                return false;
            }
            found_loopback |= row.dwLocalAddr == loopback;
        }
    }
    found_loopback
}

#[cfg(test)]
mod tests {
    use super::owns_loopback_listener;
    use std::net::TcpListener;

    #[test]
    fn recognizes_the_exact_loopback_listener_and_rejects_the_wrong_pid() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let pid = std::process::id();
        assert!(owns_loopback_listener(port, pid));
        assert!(!owns_loopback_listener(port, pid.wrapping_add(1)));
    }

    #[test]
    fn rejects_a_wildcard_listener_even_if_the_pid_matches() {
        let listener = TcpListener::bind(("0.0.0.0", 0)).unwrap();
        assert!(!owns_loopback_listener(
            listener.local_addr().unwrap().port(),
            std::process::id(),
        ));
    }
}
