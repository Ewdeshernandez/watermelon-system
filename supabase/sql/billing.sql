-- =====================================================================
-- Suscripciones y facturación del cliente (pestaña "Suscripción").
--
-- wm_subscriptions     → el plan contratado por cada cliente (clients.json id)
-- wm_invoices          → una factura por periodo (mensual o anual)
-- wm_billing_settings  → datos de transferencia por entidad, TRM manual, etc.
--
-- Solo la service role (web / cron) lee y escribe: RLS activo SIN políticas.
-- Precios por cliente y datos bancarios viven AQUÍ, nunca en el repo.
-- (Tablas nuevas: sin DROP — el editor de Supabase los bloquea.)
-- =====================================================================
create table if not exists public.wm_subscriptions (
  id                    bigint generated always as identity primary key,
  client_id             text        not null unique,
  plan                  text        not null default 'profesional'
                        check (plan in ('esencial', 'profesional', 'integral')),
  price_per_machine     numeric,                -- COP; null = precio de lista del plan
  machines              text[]      not null default '{}',
  billing_entity        text        not null default 'siga_co'
                        check (billing_entity in ('siga_co', 'wm_us')),
  cycle                 text        not null default 'mensual'
                        check (cycle in ('mensual', 'anual')),
  usd_price_per_machine numeric,                -- USD congelado 12 meses (wm_us)
  usd_locked_at         date,
  status                text        not null default 'activa'
                        check (status in ('activa', 'pausada', 'cancelada')),
  start_date            date        not null default current_date,
  next_renewal          date        not null default current_date,
  due_days              int         not null default 10,
  authorized_emails     text[]      not null default '{}',
  billing_emails        text[]      not null default '{}',
  notify_email          boolean     not null default false,
  notes                 text,
  created_at            timestamptz not null default now(),
  updated_at            timestamptz not null default now()
);

create table if not exists public.wm_invoices (
  id                    bigint generated always as identity primary key,
  number                text        not null unique,
  client_id             text        not null,
  subscription_id       bigint      not null references public.wm_subscriptions (id),
  period_start          date        not null,
  period_end            date        not null,
  issued_at             timestamptz not null default now(),
  due_date              date        not null,
  billing_entity        text        not null,
  currency              text        not null check (currency in ('COP', 'USD')),
  plan                  text        not null,
  machines              int         not null,
  unit_price            numeric     not null,
  volume_discount_pct   numeric     not null default 0,
  entity_discount_pct   numeric     not null default 0,
  months                int         not null default 1,
  months_charged        int         not null default 1,
  subtotal              numeric     not null,
  amount                numeric     not null,
  trm                   numeric,
  status                text        not null default 'pendiente'
                        check (status in ('pendiente', 'pagada', 'vencida', 'anulada')),
  paid_at               timestamptz,
  paid_method           text,
  paid_ref              text,
  provider              text        not null default 'manual',
  provider_ref          text,
  notes                 text,
  created_by            text,
  unique (subscription_id, period_start)
);

create index if not exists wm_invoices_client_idx
  on public.wm_invoices (client_id, period_start desc);

create table if not exists public.wm_billing_settings (
  key         text primary key,
  value       jsonb       not null default '{}'::jsonb,
  updated_at  timestamptz not null default now()
);

alter table public.wm_subscriptions    enable row level security;
alter table public.wm_invoices         enable row level security;
alter table public.wm_billing_settings enable row level security;
