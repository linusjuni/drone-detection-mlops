<script lang="ts">
    import './layout.css';

    let { children } = $props();
</script>

<svelte:head>
    <title>Bird vs. Drone</title>
    <link rel="icon" href="/favicon.ico" />
</svelte:head>

{@render children()}
