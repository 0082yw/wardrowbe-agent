'use client';
import { createContext } from 'react';

export const writeOperations = ['create', 'update', 'delete', 'archive', 'restore', 'rotate'] as const;
export type WriteOperation = typeof writeOperations[number];
export function permitsOperation(access: ApplicationAccessState, operation: string): boolean {
  return access.businessWrite && !!access.writeOperations?.some(value => value === operation);
}
export interface ApplicationAccessState {
  businessWrite: boolean;
  writeOperations?: readonly WriteOperation[];
  reason: string | null;
}
export const ApplicationAccessContext = createContext<ApplicationAccessState>({
  businessWrite: false,
  writeOperations: [],
  reason: null,
});
