---
title: VmbCPP ConditionHelper Class
description: Helper class for managing read/write locks using conditions, intended
  for internal use in the VmbCPP library implementation.
source: src/cameras/include/VimbaX_2023-4/api/source/VmbCPP/ConditionHelper.h
tags:
- threading
- synchronization
- vimba-sdk
- camera-api
- cpp
related: []
last_analyzed: '2026-03-09T07:54:06Z'
---

# VmbCPP ConditionHelper Class

This header file defines the ConditionHelper class within the VmbCPP namespace, which is part of the Allied Vision Vimba X camera SDK. The class provides a mechanism for managing concurrent read and write access to shared resources using condition variables. It supports both read locks (allowing multiple readers) and write locks (exclusive access), with an option for exclusive write access that bypasses normal locking. The implementation tracks the number of active readers and writing state through private member variables.

**Key concepts:** `read-write locking`, `condition variables`, `exclusive write access`, `thread synchronization`, `mutex management`

## Exports

| Name | Kind | Description |
|------|------|-------------|
| `class ConditionHelper` | class | Helper class for managing read/write lock conditions with support for exclusive write access |
| `ConditionHelper()` | method | Default constructor for ConditionHelper |
| `bool EnterReadLock(BasicLockable &rLockable)` | method | Waits until writing access has finished and acquires read lock. Returns false immediately if exclusive writing access was granted. |
| `bool EnterReadLock(MutexPtr &pMutex)` | method | Overloaded version that accepts a MutexPtr instead of BasicLockable reference |
| `void ExitReadLock(BasicLockable &rLockable)` | method | Releases a previously acquired read lock |
| `void ExitReadLock(MutexPtr &pMutex)` | method | Overloaded version that accepts a MutexPtr instead of BasicLockable reference |
| `bool EnterWriteLock(BasicLockable &rLockable, bool bExclusive = false)` | method | Waits until reading and writing access have finished and acquires write lock. If bExclusive is true, grants exclusive access. |
| `bool EnterWriteLock(MutexPtr &pMutex, bool bExclusive = false)` | method | Overloaded version that accepts a MutexPtr instead of BasicLockable reference |
| `void ExitWriteLock(BasicLockable &rLockable)` | method | Releases a previously acquired write lock |
| `void ExitWriteLock(MutexPtr &pMutex)` | method | Overloaded version that accepts a MutexPtr instead of BasicLockable reference |
