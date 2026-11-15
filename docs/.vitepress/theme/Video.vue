<script setup lang="ts">
// 课程视频的所有者禁用了站外嵌入播放，所以这里显示缩略图卡片，点击后在 YouTube 打开。
const props = defineProps<{ id: string; t?: number | string; title?: string }>()
const start = Number(props.t || 0)
const href = `https://www.youtube.com/watch?v=${props.id}${start ? `&t=${start}s` : ''}&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV`
const thumb = `https://i.ytimg.com/vi/${props.id}/hqdefault.jpg`
</script>
<template>
  <a class="video-card" :href="href" target="_blank" rel="noopener">
    <span class="video-thumb">
      <img :src="thumb" :alt="title || 'YouTube 视频'" loading="lazy" />
      <span class="video-play" aria-hidden="true">▶</span>
    </span>
    <span class="video-meta">
      <span class="video-title">🎬 {{ title || '观看原课视频' }}</span>
      <span class="video-hint">在 YouTube 打开（原视频不允许站外嵌入播放）↗</span>
    </span>
  </a>
</template>
