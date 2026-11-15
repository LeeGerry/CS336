<script setup lang="ts">
import { withBase } from 'vitepress'
const props = defineProps<{ src: string; title?: string; height?: number | string }>()
</script>
<template>
  <figure class="demo-frame">
    <figcaption v-if="title">▶ 交互演示：{{ title }} <a :href="withBase('/demos/' + src)" target="_blank">全屏打开 ↗</a></figcaption>
    <iframe :src="withBase('/demos/' + src)" :style="{ height: (height || 520) + 'px' }" loading="lazy"></iframe>
  </figure>
</template>
